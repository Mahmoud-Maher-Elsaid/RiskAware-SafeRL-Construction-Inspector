from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from scripts.generate_causal_expert_demonstrations import generate
from scripts.train_behavior_cloning_v2 import DemonstrationSequences, build_mmap_cache


def test_causal_dataset_is_observation_only_mask_valid_and_split_safe(
    tmp_path: Path,
) -> None:
    output = tmp_path / "causal"
    summary = generate(
        output,
        target_transitions=80,
        chunk_size=40,
        seeds=12,
        minimum_unique_seeds=10,
    )
    assert summary["transitions"] >= 80
    assert summary["observation_only_contract"]
    assert not summary["hidden_state_stored_or_consumed"]
    assert summary["invalid_actions"] == 0
    assert summary["episode_split_integrity"]
    episode_splits: dict[str, set[str]] = {}
    stored = 0
    allowed = {
        "maps",
        "states",
        "action_masks",
        "next_maps",
        "next_states",
        "causal_expert_proposed_actions",
        "causal_expert_executed_actions",
        "shield_decisions",
        "rewards",
        "safety_costs",
        "terminated",
        "truncated",
        "success",
        "mission_progress",
        "episode_ids",
        "episode_steps",
        "splits",
        "discovered_target_counts",
        "target_types",
    }
    for chunk in summary["chunks"]:
        path = Path(chunk["path"])
        assert hashlib.sha256(path.read_bytes()).hexdigest() == chunk["sha256"]
        with np.load(path, allow_pickle=False) as payload:
            assert set(payload.files) == allowed
            actions = payload["causal_expert_executed_actions"].astype(np.int64)
            assert np.all(payload["action_masks"][np.arange(len(actions)), actions] == 1)
            for episode, split in zip(payload["episode_ids"], payload["splits"], strict=True):
                episode_splits.setdefault(str(episode), set()).add(str(split))
            stored += len(actions)
    assert stored == summary["transitions"]
    assert all(len(splits) == 1 for splits in episode_splits.values())


def test_generator_refuses_original_dataset_path() -> None:
    with np.testing.assert_raises(ValueError):
        generate(
            Path("artifacts/strong_policy_upgrade/expert_demonstrations"),
            target_transitions=1,
            chunk_size=1,
            seeds=1,
            minimum_unique_seeds=1,
        )


def test_manifest_metadata_is_json_serializable(tmp_path: Path) -> None:
    summary = generate(
        tmp_path / "causal",
        target_transitions=20,
        chunk_size=20,
        seeds=10,
        minimum_unique_seeds=1,
    )
    manifest = dict(summary)
    manifest.pop("episode_summaries")
    assert json.loads(json.dumps(manifest))["action_key"] == ("causal_expert_executed_actions")
    root = tmp_path / "causal"
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    cache = build_mmap_cache(root)
    assert cache["action_key"] == "causal_expert_executed_actions"
    dataset = DemonstrationSequences(root, "train", 8)
    if len(dataset):
        batch = dataset[0]
        assert np.all(batch["masks"].gather(-1, batch["actions"].unsqueeze(-1)).squeeze(-1).numpy())
