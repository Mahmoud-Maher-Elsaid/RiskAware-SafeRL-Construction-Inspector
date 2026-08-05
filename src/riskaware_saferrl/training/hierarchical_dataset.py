from __future__ import annotations

import hashlib
import json
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

HIERARCHICAL_ARRAYS = (
    "maps",
    "states",
    "option_masks",
    "options",
    "target_coordinates",
    "option_durations",
    "shield_interventions",
    "mission_progress",
    "rewards",
    "vector_costs",
    "episode_ids",
    "episode_steps",
    "splits",
    "recurrent_context",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build_hierarchical_mmap_cache(dataset_dir: Path) -> dict[str, Any]:
    manifest = json.loads((dataset_dir / "manifest.json").read_text(encoding="utf-8"))
    cache_dir = dataset_dir / "mmap_cache"
    cache_manifest_path = cache_dir / "cache_manifest.json"
    if cache_manifest_path.exists():
        cached = json.loads(cache_manifest_path.read_text(encoding="utf-8"))
        if cached.get("source_dataset_sha256") == manifest["dataset_sha256"] and all(
            Path(item["path"]).exists() for item in cached["arrays"]
        ):
            return cached
    cache_dir.mkdir(parents=True, exist_ok=True)
    arrays: list[dict[str, Any]] = []
    for chunk_index, source in enumerate(manifest["chunks"]):
        source_path = Path(source["path"])
        with np.load(source_path, allow_pickle=False) as archive:
            for name in HIERARCHICAL_ARRAYS:
                output = cache_dir / f"chunk_{chunk_index:05d}_{name}.npy"
                np.save(output, archive[name], allow_pickle=False)
                arrays.append(
                    {
                        "chunk": chunk_index,
                        "name": name,
                        "path": output.as_posix(),
                        "sha256": _sha256(output),
                    }
                )
    result = {
        "status": "PASSED",
        "source_dataset_sha256": manifest["dataset_sha256"],
        "source_chunks": manifest["chunks"],
        "arrays": arrays,
    }
    cache_manifest_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


@dataclass(frozen=True)
class SequenceIndex:
    chunk: int
    start: int
    stop: int
    split: str
    episode_id: str


class _ChunkLRU:
    def __init__(self, cache: dict[str, Any], maximum: int = 2) -> None:
        self.maximum = maximum
        self.paths = {
            (int(item["chunk"]), str(item["name"])): Path(item["path"]) for item in cache["arrays"]
        }
        self.active: OrderedDict[int, dict[str, np.ndarray]] = OrderedDict()

    def get(self, chunk: int) -> dict[str, np.ndarray]:
        if chunk in self.active:
            self.active.move_to_end(chunk)
            return self.active[chunk]
        values = {
            name: np.load(self.paths[(chunk, name)], mmap_mode="r", allow_pickle=False)
            for name in HIERARCHICAL_ARRAYS
        }
        self.active[chunk] = values
        while len(self.active) > self.maximum:
            self.active.popitem(last=False)
        return values


class HierarchicalSequenceDataset(Dataset[dict[str, torch.Tensor]]):
    """Lazy recurrent sequences backed by a two-chunk mmap LRU."""

    def __init__(
        self,
        dataset_dir: Path,
        *,
        split: str,
        sequence_length: int = 8,
        stride: int = 4,
        max_cached_chunks: int = 2,
    ) -> None:
        self.dataset_dir = dataset_dir
        self.split = split
        self.sequence_length = sequence_length
        self.cache_metadata = build_hierarchical_mmap_cache(dataset_dir)
        self._chunks = _ChunkLRU(self.cache_metadata, max_cached_chunks)
        self.sequences: list[SequenceIndex] = []
        chunk_ids = sorted({int(item["chunk"]) for item in self.cache_metadata["arrays"]})
        for chunk in chunk_ids:
            arrays = self._chunks.get(chunk)
            episode_ids = arrays["episode_ids"]
            splits = arrays["splits"]
            start = 0
            while start < len(episode_ids):
                stop = start + 1
                while stop < len(episode_ids) and episode_ids[stop] == episode_ids[start]:
                    stop += 1
                if str(splits[start]) == split:
                    for sequence_start in range(
                        start, max(start + 1, stop - sequence_length + 1), stride
                    ):
                        sequence_stop = min(sequence_start + sequence_length, stop)
                        if sequence_stop - sequence_start == sequence_length:
                            self.sequences.append(
                                SequenceIndex(
                                    chunk,
                                    sequence_start,
                                    sequence_stop,
                                    split,
                                    str(episode_ids[start]),
                                )
                            )
                start = stop

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        sequence = self.sequences[index]
        arrays = self._chunks.get(sequence.chunk)
        selected = slice(sequence.start, sequence.stop)
        states = np.asarray(arrays["states"][selected], dtype=np.float32)
        context = np.asarray(arrays["recurrent_context"][selected], dtype=np.float32)
        extras = np.stack(
            (
                np.asarray(arrays["mission_progress"][selected], dtype=np.float32),
                np.asarray(arrays["shield_interventions"][selected], dtype=np.float32),
                np.asarray(arrays["option_durations"][selected], dtype=np.float32) / 8.0,
            ),
            axis=-1,
        )
        structured = np.concatenate((states, context.reshape(len(states), -1), extras), axis=-1)
        coordinates = np.asarray(arrays["target_coordinates"][selected], dtype=np.int64)
        targets = np.where(
            coordinates[:, 0] >= 0,
            coordinates[:, 0] * 16 + coordinates[:, 1],
            256,
        )

        def writable(value: np.ndarray, dtype: np.dtype[Any] | type[Any]) -> np.ndarray:
            return np.array(value, dtype=dtype, order="C", copy=True)

        return {
            "maps": torch.from_numpy(writable(arrays["maps"][selected], np.float32)),
            "states": torch.from_numpy(writable(structured, np.float32)),
            "option_masks": torch.from_numpy(writable(arrays["option_masks"][selected], np.bool_)),
            "options": torch.from_numpy(writable(arrays["options"][selected], np.int64)),
            "targets": torch.from_numpy(writable(targets, np.int64)),
            "durations": torch.from_numpy(
                writable(arrays["option_durations"][selected], np.int64) - 1
            ),
            "rewards": torch.from_numpy(writable(arrays["rewards"][selected], np.float32)),
            "vector_costs": torch.from_numpy(
                writable(arrays["vector_costs"][selected], np.float32)
            ),
        }
