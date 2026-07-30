from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from riskaware_saferrl.hierarchical.schemas import MissionOption
from riskaware_saferrl.training.hierarchical_dataset import (
    HierarchicalSequenceDataset,
    SequenceIndex,
)
from riskaware_saferrl.training.hierarchical_demonstrations import (
    HierarchicalDatasetBuilder,
    HierarchicalDerivationConfig,
    derive_option,
    sha256_file,
)


def semantic_map() -> np.ndarray:
    value = np.zeros((11, 16, 16), dtype=np.uint8)
    value[5, 4, 4] = 1
    value[9, :8, :8] = 1
    return value


def create_source(root: Path) -> Path:
    source = root / "source"
    chunks = source / "chunks"
    chunks.mkdir(parents=True)
    rows = 16
    maps = np.stack([semantic_map() for _ in range(rows)])
    maps[4, 8, 4, 5] = 1
    maps[8, 2, 4, 5] = 1
    path = chunks / "causal_demonstrations_00000.npz"
    np.savez_compressed(
        path,
        maps=maps,
        states=np.zeros((rows, 17), dtype=np.float16),
        action_masks=np.ones((rows, 5), dtype=np.uint8),
        causal_expert_executed_actions=np.asarray(
            [3, 3, 3, 4, 4, 3, 3, 3, 0, 0, 1, 1, 2, 2, 3, 3], dtype=np.int8
        ),
        shield_decisions=np.asarray(["accept"] * rows),
        rewards=np.ones(rows, dtype=np.float32),
        safety_costs=np.zeros(rows, dtype=np.float32),
        terminated=np.zeros(rows, dtype=np.bool_),
        truncated=np.zeros(rows, dtype=np.bool_),
        success=np.zeros(rows, dtype=np.bool_),
        mission_progress=np.linspace(0, 1, rows, dtype=np.float32),
        episode_ids=np.asarray(["train-1"] * 8 + ["test-2"] * 8),
        episode_steps=np.asarray(list(range(8)) * 2, dtype=np.int16),
        splits=np.asarray(["train"] * 8 + ["test"] * 8),
    )
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    (source / "manifest.json").write_text(
        json.dumps(
            {
                "dataset_sha256": "source-dataset",
                "chunks": [{"path": path.as_posix(), "sha256": digest, "rows": rows}],
            }
        ),
        encoding="utf-8",
    )
    return source


def test_labels_use_only_observation_and_identify_inspection() -> None:
    value = semantic_map()
    value[8, 4, 5] = 1
    option, target, _ = derive_option(
        value,
        np.ones(5, dtype=np.uint8),
        4,
        previous_action=3,
        shield_intervened=False,
    )
    assert option == MissionOption.INSPECT_PPE_VIOLATION
    assert target == (4, 5)


def test_builder_is_deterministic_preserves_source_and_prevents_leakage(
    tmp_path: Path,
) -> None:
    source = create_source(tmp_path)
    source_chunk = next((source / "chunks").glob("*.npz"))
    before = sha256_file(source_chunk)
    first = HierarchicalDatasetBuilder(
        source,
        tmp_path / "output-a",
        HierarchicalDerivationConfig(decision_interval=4, records_per_chunk=2),
    ).build()
    second = HierarchicalDatasetBuilder(
        source,
        tmp_path / "output-b",
        HierarchicalDerivationConfig(decision_interval=4, records_per_chunk=2),
    ).build()
    assert first["dataset_sha256"] == second["dataset_sha256"]
    assert first["episode_split_integrity"]
    assert first["split_counts"] == {"train": 2, "test": 2}
    assert sha256_file(source_chunk) == before
    assert first["source_chunk_hashes_unchanged"]


def test_record_schema_has_valid_options_targets_and_primitives(tmp_path: Path) -> None:
    source = create_source(tmp_path)
    output = tmp_path / "output"
    manifest = HierarchicalDatasetBuilder(source, output).build()
    with np.load(Path(manifest["chunks"][0]["path"]), allow_pickle=False) as data:
        assert np.all((data["options"] >= 0) & (data["options"] < len(MissionOption)))
        assert data["target_coordinates"].shape[1:] == (2,)
        assert data["executed_primitives"].shape[1:] == (4,)
        assert np.all(data["executed_primitives"][data["executed_primitive_masks"]] < 5)
        assert data["vector_costs"].shape[1:] == (7,)
        episode_to_split: dict[str, str] = {}
        for episode_id, split in zip(data["episode_ids"], data["splits"], strict=True):
            assert episode_to_split.setdefault(str(episode_id), str(split)) == str(split)


def test_lazy_sequence_index_contains_only_metadata_and_is_deterministic(
    tmp_path: Path,
) -> None:
    source = create_source(tmp_path)
    output = tmp_path / "output"
    HierarchicalDatasetBuilder(source, output).build()
    dataset = HierarchicalSequenceDataset(output, split="train", sequence_length=2, stride=1)
    assert dataset.sequences
    assert all(isinstance(item, SequenceIndex) for item in dataset.sequences)
    first = dataset[0]
    second = dataset[0]
    assert first["states"].shape[-1] == 32
    assert all(torch.equal(first[key], second[key]) for key in first)
    assert len(dataset._chunks.active) <= 2
