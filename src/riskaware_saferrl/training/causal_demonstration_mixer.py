from __future__ import annotations

import hashlib
import json
from collections.abc import Iterator
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

import numpy as np

MixMode = Literal[
    "causal_only",
    "causal_plus_privileged_pretraining",
    "causal_plus_privileged_distillation_with_confidence_filter",
]


@dataclass(frozen=True)
class TrainingStage:
    source: Literal["causal", "privileged"]
    dataset_dir: str
    split: str
    base_weight: float
    confidence_filter: bool


@dataclass(frozen=True)
class WeightedSampleReference:
    source: Literal["causal", "privileged"]
    chunk_index: int
    row: int
    split: str
    label: int
    weight: float
    confidence: float
    fingerprint: str


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def observation_fingerprint(
    semantic_map: np.ndarray,
    state: np.ndarray,
    action_mask: np.ndarray,
) -> str:
    digest = hashlib.blake2b(digest_size=16)
    digest.update(np.ascontiguousarray(semantic_map).tobytes())
    digest.update(np.ascontiguousarray(state).tobytes())
    digest.update(np.ascontiguousarray(action_mask).tobytes())
    return digest.hexdigest()


def privileged_observability_confidence(
    semantic_map: np.ndarray,
    action_mask: np.ndarray,
    label: int,
) -> float:
    mask = np.asarray(action_mask, dtype=np.bool_)
    if label < 0 or label >= len(mask) or not mask[label]:
        return 0.0
    if label == 4:
        return 1.0 if mask[4] else 0.0
    remaining_detected = (semantic_map[1] > 0) & (semantic_map[10] == 0)
    if np.any(remaining_detected):
        return 0.9
    visible_unvisited = (semantic_map[9] > 0) & (semantic_map[4] == 0)
    if np.any(visible_unvisited):
        return 0.6
    return 0.25


class CausalDemonstrationMixer:
    """Build reproducible causal-primary plans without mutating either source."""

    def __init__(
        self,
        causal_dataset_dir: Path,
        privileged_dataset_dir: Path,
        *,
        confidence_threshold: float = 0.6,
        privileged_weight: float = 0.25,
    ) -> None:
        self.causal_dataset_dir = causal_dataset_dir
        self.privileged_dataset_dir = privileged_dataset_dir
        self.confidence_threshold = confidence_threshold
        self.privileged_weight = privileged_weight
        self.causal_manifest = self._load_manifest(causal_dataset_dir)
        self.privileged_manifest = self._load_manifest(privileged_dataset_dir)
        self._source_hashes = self.source_hashes()

    @staticmethod
    def _load_manifest(dataset_dir: Path) -> dict:
        path = dataset_dir / "manifest.json"
        if not path.is_file():
            raise FileNotFoundError(f"Dataset manifest not found: {path}")
        return json.loads(path.read_text(encoding="utf-8"))

    @staticmethod
    def _action_key(manifest: dict) -> str:
        return manifest.get("action_key", "teacher_executed_actions")

    def source_hashes(self) -> dict[str, dict[str, str]]:
        return {
            "causal": {
                "manifest": _sha256_file(self.causal_dataset_dir / "manifest.json"),
                **{
                    Path(chunk["path"]).name: _sha256_file(Path(chunk["path"]))
                    for chunk in self.causal_manifest["chunks"]
                },
            },
            "privileged": {
                "manifest": _sha256_file(self.privileged_dataset_dir / "manifest.json"),
                **{
                    Path(chunk["path"]).name: _sha256_file(Path(chunk["path"]))
                    for chunk in self.privileged_manifest["chunks"]
                },
            },
        }

    def assert_sources_unchanged(self) -> None:
        if self.source_hashes() != self._source_hashes:
            raise RuntimeError("A source demonstration dataset changed during mixing")

    def stages(self, mode: MixMode) -> list[TrainingStage]:
        causal = TrainingStage(
            source="causal",
            dataset_dir=self.causal_dataset_dir.as_posix(),
            split="train",
            base_weight=1.0,
            confidence_filter=False,
        )
        privileged = TrainingStage(
            source="privileged",
            dataset_dir=self.privileged_dataset_dir.as_posix(),
            split="train",
            base_weight=self.privileged_weight,
            confidence_filter=mode.endswith("confidence_filter"),
        )
        if mode == "causal_only":
            return [causal]
        if mode in {
            "causal_plus_privileged_pretraining",
            "causal_plus_privileged_distillation_with_confidence_filter",
        }:
            return [privileged, causal]
        raise ValueError(f"Unknown mix mode: {mode}")

    def evaluation_sources(self) -> dict[str, Literal["causal"]]:
        return {"validation": "causal", "test": "causal"}

    def _causal_label_index(self) -> tuple[dict[str, int], int]:
        labels: dict[str, int] = {}
        conflicts = 0
        action_key = self._action_key(self.causal_manifest)
        for chunk in self.causal_manifest["chunks"]:
            with np.load(chunk["path"], allow_pickle=False) as payload:
                maps = payload["maps"]
                states = payload["states"]
                masks = payload["action_masks"]
                actions = payload[action_key]
                selected = np.flatnonzero(payload["splits"] == "train")
                for row in selected:
                    fingerprint = observation_fingerprint(
                        maps[row],
                        states[row],
                        masks[row],
                    )
                    label = int(actions[row])
                    previous = labels.get(fingerprint)
                    if previous is not None and previous != label:
                        labels[fingerprint] = -1
                        conflicts += 1
                    elif previous is None:
                        labels[fingerprint] = label
        return labels, conflicts

    def iter_causal_samples(self, split: str) -> Iterator[WeightedSampleReference]:
        action_key = self._action_key(self.causal_manifest)
        for chunk_index, chunk in enumerate(self.causal_manifest["chunks"]):
            with np.load(chunk["path"], allow_pickle=False) as payload:
                maps = payload["maps"]
                states = payload["states"]
                masks = payload["action_masks"]
                actions = payload[action_key]
                selected = np.flatnonzero(payload["splits"] == split)
                for row in selected:
                    yield WeightedSampleReference(
                        source="causal",
                        chunk_index=chunk_index,
                        row=int(row),
                        split=split,
                        label=int(actions[row]),
                        weight=1.0,
                        confidence=1.0,
                        fingerprint=observation_fingerprint(
                            maps[row],
                            states[row],
                            masks[row],
                        ),
                    )

    def build_distillation_index(self, output_path: Path) -> dict:
        causal_labels, causal_internal_conflicts = self._causal_label_index()
        action_key = self._action_key(self.privileged_manifest)
        accepted = 0
        rejected_low_confidence = 0
        rejected_conflict = 0
        rejected_invalid = 0
        output_path.parent.mkdir(parents=True, exist_ok=True)
        digest = hashlib.sha256()
        with output_path.open("w", encoding="utf-8", newline="\n") as stream:
            for chunk_index, chunk in enumerate(self.privileged_manifest["chunks"]):
                with np.load(chunk["path"], allow_pickle=False) as payload:
                    maps = payload["maps"]
                    states = payload["states"]
                    masks = payload["action_masks"]
                    actions = payload[action_key]
                    selected = np.flatnonzero(payload["splits"] == "train")
                    for row in selected:
                        label = int(actions[row])
                        confidence = privileged_observability_confidence(
                            maps[row],
                            masks[row],
                            label,
                        )
                        if confidence == 0.0:
                            rejected_invalid += 1
                            continue
                        fingerprint = observation_fingerprint(
                            maps[row],
                            states[row],
                            masks[row],
                        )
                        causal_label = causal_labels.get(fingerprint)
                        if causal_label is not None and causal_label != label:
                            rejected_conflict += 1
                            continue
                        if confidence < self.confidence_threshold:
                            rejected_low_confidence += 1
                            continue
                        reference = WeightedSampleReference(
                            source="privileged",
                            chunk_index=chunk_index,
                            row=int(row),
                            split="train",
                            label=label,
                            weight=self.privileged_weight * confidence,
                            confidence=confidence,
                            fingerprint=fingerprint,
                        )
                        line = (
                            json.dumps(asdict(reference), sort_keys=True, separators=(",", ":"))
                            + "\n"
                        )
                        stream.write(line)
                        digest.update(line.encode())
                        accepted += 1
        self.assert_sources_unchanged()
        summary = {
            "status": "PASSED",
            "accepted_privileged_samples": accepted,
            "rejected_low_confidence": rejected_low_confidence,
            "rejected_conflict": rejected_conflict,
            "rejected_invalid": rejected_invalid,
            "causal_internal_conflicts": causal_internal_conflicts,
            "confidence_threshold": self.confidence_threshold,
            "privileged_weight": self.privileged_weight,
            "index_path": output_path.as_posix(),
            "index_sha256": digest.hexdigest(),
            "evaluation_sources": self.evaluation_sources(),
            "source_hashes_unchanged": True,
        }
        (output_path.with_suffix(".summary.json")).write_text(
            json.dumps(summary, indent=2) + "\n", encoding="utf-8"
        )
        return summary
