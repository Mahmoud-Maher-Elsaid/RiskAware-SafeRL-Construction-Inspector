from __future__ import annotations

from pathlib import Path

import numpy as np

from scripts.generate_expert_demonstrations import generate


def test_demonstrations_have_valid_masked_actions_and_seed_splits(tmp_path: Path) -> None:
    summary = generate(
        tmp_path,
        target_transitions=80,
        chunk_size=40,
        seeds=2,
        minimum_unique_seeds=1,
    )
    assert summary["transitions"] >= 80
    assert summary["unique_seeds"] >= 1
    episode_splits: dict[str, set[str]] = {}
    for chunk in summary["chunks"]:
        with np.load(chunk["path"], allow_pickle=False) as payload:
            for index, action in enumerate(payload["teacher_executed_actions"]):
                assert payload["action_masks"][index, action] == 1
            for episode, split in zip(payload["episode_ids"], payload["splits"], strict=True):
                episode_splits.setdefault(str(episode), set()).add(str(split))
    assert all(len(splits) == 1 for splits in episode_splits.values())
