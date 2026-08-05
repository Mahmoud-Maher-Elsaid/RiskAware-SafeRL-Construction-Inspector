from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from riskaware_saferrl.training.causal_demonstration_mixer import (
    CausalDemonstrationMixer,
)


def _write_dataset(
    root: Path,
    *,
    causal: bool,
    labels: list[int],
    splits: list[str],
    conflict_first_map: bool = False,
    low_confidence_last: bool = False,
) -> None:
    root.mkdir(parents=True)
    chunk_dir = root / "chunks"
    chunk_dir.mkdir()
    rows = len(labels)
    maps = np.zeros((rows, 11, 16, 16), dtype=np.uint8)
    states = np.zeros((rows, 17), dtype=np.float16)
    masks = np.ones((rows, 5), dtype=np.uint8)
    for index in range(rows):
        maps[index, 5, index + 1, index + 1] = 1
        maps[index, 9, :4, :4] = 1
        states[index, 9] = 0.5
    if conflict_first_map and rows:
        maps[0].fill(0)
        maps[0, 5, 1, 1] = 1
        maps[0, 9, :4, :4] = 1
    if low_confidence_last and rows:
        maps[-1, 4] = maps[-1, 9]
    action_key = "causal_expert_executed_actions" if causal else "teacher_executed_actions"
    path = chunk_dir / "chunk_00000.npz"
    np.savez_compressed(
        path,
        maps=maps,
        states=states,
        action_masks=masks,
        episode_ids=np.asarray(
            [f"{split}:episode:{index}" for index, split in enumerate(splits)],
            dtype="U64",
        ),
        splits=np.asarray(splits, dtype="U10"),
        **{action_key: np.asarray(labels, dtype=np.int8)},
    )
    chunk_hash = hashlib.sha256(path.read_bytes()).hexdigest()
    manifest = {
        "transitions": rows,
        "action_key": action_key,
        "dataset_sha256": hashlib.sha256(chunk_hash.encode()).hexdigest(),
        "chunks": [{"path": path.as_posix(), "rows": rows, "sha256": chunk_hash}],
    }
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


def test_modes_keep_causal_evaluation_authoritative(tmp_path: Path) -> None:
    causal = tmp_path / "causal"
    privileged = tmp_path / "privileged"
    _write_dataset(
        causal,
        causal=True,
        labels=[0, 1, 2],
        splits=["train", "validation", "test"],
    )
    _write_dataset(
        privileged,
        causal=False,
        labels=[0, 1],
        splits=["train", "test"],
    )
    mixer = CausalDemonstrationMixer(causal, privileged)
    assert [stage.source for stage in mixer.stages("causal_only")] == ["causal"]
    assert [stage.source for stage in mixer.stages("causal_plus_privileged_pretraining")] == [
        "privileged",
        "causal",
    ]
    assert mixer.evaluation_sources() == {
        "validation": "causal",
        "test": "causal",
    }


def test_conflicts_and_low_confidence_are_filtered_reproducibly(
    tmp_path: Path,
) -> None:
    causal = tmp_path / "causal"
    privileged = tmp_path / "privileged"
    _write_dataset(
        causal,
        causal=True,
        labels=[0],
        splits=["train"],
    )
    _write_dataset(
        privileged,
        causal=False,
        labels=[1, 2],
        splits=["train", "train"],
        conflict_first_map=True,
        low_confidence_last=True,
    )
    mixer = CausalDemonstrationMixer(
        causal,
        privileged,
        confidence_threshold=0.6,
    )
    first = mixer.build_distillation_index(tmp_path / "first.jsonl")
    second = mixer.build_distillation_index(tmp_path / "second.jsonl")
    assert first["rejected_conflict"] == 1
    assert first["rejected_low_confidence"] == 1
    assert first["accepted_privileged_samples"] == 0
    assert first["index_sha256"] == second["index_sha256"]


def test_causal_samples_have_unit_weight_and_no_split_leakage(
    tmp_path: Path,
) -> None:
    causal = tmp_path / "causal"
    privileged = tmp_path / "privileged"
    _write_dataset(
        causal,
        causal=True,
        labels=[0, 1, 2],
        splits=["train", "validation", "test"],
    )
    _write_dataset(
        privileged,
        causal=False,
        labels=[3],
        splits=["train"],
    )
    mixer = CausalDemonstrationMixer(causal, privileged)
    train = list(mixer.iter_causal_samples("train"))
    test = list(mixer.iter_causal_samples("test"))
    assert all(sample.source == "causal" and sample.weight == 1.0 for sample in train)
    assert all(sample.source == "causal" for sample in test)
    assert {sample.fingerprint for sample in train}.isdisjoint(
        sample.fingerprint for sample in test
    )


def test_mixing_does_not_modify_sources(tmp_path: Path) -> None:
    causal = tmp_path / "causal"
    privileged = tmp_path / "privileged"
    _write_dataset(causal, causal=True, labels=[0], splits=["train"])
    _write_dataset(privileged, causal=False, labels=[0], splits=["train"])
    before = {
        path: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in (causal / "chunks").glob("*.npz")
    } | {
        path: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in (privileged / "chunks").glob("*.npz")
    }
    mixer = CausalDemonstrationMixer(causal, privileged)
    mixer.build_distillation_index(tmp_path / "index.jsonl")
    after = {path: hashlib.sha256(path.read_bytes()).hexdigest() for path in before}
    assert before == after
