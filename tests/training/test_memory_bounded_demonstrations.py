from __future__ import annotations

import gc
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
import psutil
import torch
from torch.utils.data import DataLoader

from scripts.generate_expert_demonstrations import generate
from scripts.train_behavior_cloning_v2 import (
    DemonstrationSequences,
    SequenceIndex,
    build_mmap_cache,
)


def _dataset(tmp_path: Path, transitions: int = 192) -> Path:
    root = tmp_path / "demonstrations"
    summary = generate(
        root,
        target_transitions=transitions,
        chunk_size=96,
        seeds=12,
        minimum_unique_seeds=10,
    )
    manifest = dict(summary)
    manifest.pop("episode_summaries")
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return root


def _source_hashes(root: Path) -> dict[str, str]:
    return {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted((root / "chunks").glob("*.npz"))
    }


def test_construction_keeps_only_lightweight_sequence_metadata(tmp_path: Path) -> None:
    root = _dataset(tmp_path)
    dataset = DemonstrationSequences(root, "train", 8)
    assert dataset.sequences
    assert isinstance(dataset.sequences[0], SequenceIndex)
    assert not any(
        isinstance(value, np.ndarray)
        for record in dataset.sequences
        for value in (
            record.chunk,
            record.start,
            record.stop,
            record.split,
            record.episode_id,
        )
    )
    assert dataset.chunk_cache.active_chunks <= 2


def test_index_memory_growth_is_bounded_by_metadata(tmp_path: Path) -> None:
    root = _dataset(tmp_path, transitions=384)
    process = psutil.Process()
    gc.collect()
    before = process.memory_info().rss
    dataset = DemonstrationSequences(root, "train", 4)
    gc.collect()
    growth = process.memory_info().rss - before
    source_array_bytes = sum(path.stat().st_size for path in (root / "mmap_cache").rglob("*.npy"))
    assert growth < max(32 * 1024**2, source_array_bytes // 2)
    assert all(isinstance(record, SequenceIndex) for record in dataset.sequences)


def test_identical_requests_are_deterministic_and_episode_bounded(tmp_path: Path) -> None:
    root = _dataset(tmp_path)
    dataset = DemonstrationSequences(root, "train", 8)
    first = dataset[0]
    second = dataset[0]
    for name in first:
        torch.testing.assert_close(first[name], second[name])
    record = dataset.sequences[0]
    arrays = dataset.chunk_cache.get(record.chunk)
    assert np.unique(arrays["episode_ids"][record.start : record.stop]).size == 1
    assert np.all(arrays["splits"][record.start : record.stop] == "train")


def test_no_split_leakage_and_all_teacher_actions_are_valid(tmp_path: Path) -> None:
    root = _dataset(tmp_path)
    episode_splits: dict[str, set[str]] = {}
    for split in ("train", "validation", "test"):
        dataset = DemonstrationSequences(root, split, 8)
        for record in dataset.sequences:
            episode_splits.setdefault(record.episode_id, set()).add(record.split)
            batch = dataset[dataset.sequences.index(record)]
            assert torch.all(batch["masks"].gather(-1, batch["actions"].unsqueeze(-1)).squeeze(-1))
    assert all(len(splits) == 1 for splits in episode_splits.values())


def test_source_hashes_unchanged_and_cache_rebuild_is_deterministic(tmp_path: Path) -> None:
    root = _dataset(tmp_path)
    source_hashes = _source_hashes(root)
    first = build_mmap_cache(root)
    first_hashes = {
        path.relative_to(root / "mmap_cache").as_posix(): hashlib.sha256(
            path.read_bytes()
        ).hexdigest()
        for path in sorted((root / "mmap_cache").rglob("*.npy"))
    }
    shutil.rmtree(root / "mmap_cache")
    second = build_mmap_cache(root)
    second_hashes = {
        path.relative_to(root / "mmap_cache").as_posix(): hashlib.sha256(
            path.read_bytes()
        ).hexdigest()
        for path in sorted((root / "mmap_cache").rglob("*.npy"))
    }
    assert first == second
    assert first_hashes == second_hashes
    assert _source_hashes(root) == source_hashes


def test_dataloader_1000_batches_has_stable_rss(tmp_path: Path) -> None:
    root = _dataset(tmp_path)
    dataset = DemonstrationSequences(root, "train", 8)
    loader = DataLoader(dataset, batch_size=2, num_workers=0, pin_memory=False)
    process = psutil.Process()
    baseline = process.memory_info().rss
    peak = baseline
    iterator = iter(loader)
    for _ in range(1000):
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            batch = next(iterator)
        assert torch.all(batch["masks"].gather(-1, batch["actions"].unsqueeze(-1)).squeeze(-1))
        peak = max(peak, process.memory_info().rss)
    assert peak - baseline < 128 * 1024**2
