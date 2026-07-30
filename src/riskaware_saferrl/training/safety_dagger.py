from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from riskaware_saferrl.safety import EventAwareShieldDecisionV3, SafetyVectorCost


@dataclass(frozen=True)
class SafetyDaggerSelection:
    selected: bool
    reasons: tuple[str, ...]
    priority: float


def select_safety_state(
    *,
    policy_action: int,
    teacher_action: int,
    policy_shield: EventAwareShieldDecisionV3,
    recent_positions: tuple[tuple[int, int], ...],
    include_safe_example: bool,
) -> SafetyDaggerSelection:
    """Select observation-derived risk states without simulator-private data."""
    proposed = policy_shield.proposed_vector_cost
    reasons: list[str] = []
    if policy_action != teacher_action:
        reasons.append("teacher_disagreement")
    if policy_shield.shield_decision != "accept":
        reasons.append("shield_intervention")
    if policy_shield.emergency_stop:
        reasons.append("emergency_stop")
    if proposed.collision > 0:
        reasons.append("pre_collision_risk")
    if proposed.restricted_zone > 0:
        reasons.append("pre_restricted_zone_risk")
    if proposed.worker_near_miss > 0:
        reasons.append("pre_worker_near_miss")
    if proposed.uncontrolled_semantic_risk > 0:
        reasons.append("semantic_risk_exposure")
    if proposed.uncertainty > 0:
        reasons.append("uncertainty_exposure")
    if len(recent_positions) >= 8 and len(set(recent_positions[-8:])) <= 2:
        reasons.append("deadlock")
    if not reasons and include_safe_example:
        reasons.append("task_anchor")
    hard = proposed.collision + proposed.restricted_zone + proposed.worker_near_miss
    soft = proposed.uncontrolled_semantic_risk + proposed.uncertainty
    priority = 10.0 * hard + soft + 2.0 * (policy_action != teacher_action)
    return SafetyDaggerSelection(bool(reasons), tuple(reasons), float(priority))


class SafetyCorrectionWriter:
    """Immutable, bounded-memory NPZ writer for policy-visited corrections."""

    def __init__(self, directory: Path, chunk_size: int = 4_000) -> None:
        self.directory = directory
        self.chunk_size = chunk_size
        self.records: list[dict[str, Any]] = []
        self.chunks: list[dict[str, Any]] = []
        self.action_counts = np.zeros(5, dtype=np.int64)
        directory.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _sha256(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
        return digest.hexdigest()

    def add(self, record: dict[str, Any]) -> None:
        action = int(record["teacher_action"])
        if not bool(np.asarray(record["action_mask"])[action]):
            raise ValueError("Refusing to store an invalid safety-teacher action")
        self.records.append(record)
        self.action_counts[action] += 1
        if len(self.records) >= self.chunk_size:
            self.flush()

    def flush(self) -> None:
        if not self.records:
            return
        path = self.directory / f"safety_corrections_{len(self.chunks):05d}.npz"
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite immutable safety chunk: {path}")
        arrays = {
            "maps": np.stack([row["map"] for row in self.records]).astype(np.uint8),
            "states": np.stack([row["state"] for row in self.records]).astype(np.float16),
            "action_masks": np.stack([row["action_mask"] for row in self.records]).astype(np.uint8),
            "causal_expert_executed_actions": np.asarray(
                [row["teacher_action"] for row in self.records], dtype=np.int8
            ),
            "policy_actions": np.asarray(
                [row["policy_action"] for row in self.records], dtype=np.int8
            ),
            "shield_actions": np.asarray(
                [row["shield_action"] for row in self.records], dtype=np.int8
            ),
            "episode_ids": np.asarray([row["episode_id"] for row in self.records], dtype="U128"),
            "splits": np.asarray(["train"] * len(self.records), dtype="U10"),
            "selection_reasons": np.asarray(
                ["|".join(row["selection_reasons"]) for row in self.records], dtype="U256"
            ),
            "selection_priorities": np.asarray(
                [row["selection_priority"] for row in self.records], dtype=np.float32
            ),
            "predicted_vector_costs": np.asarray(
                [
                    SafetyVectorCost(**row["predicted_vector_cost"]).as_tuple()
                    for row in self.records
                ],
                dtype=np.float32,
            ),
        }
        temporary = path.with_suffix(".npz.tmp")
        with temporary.open("wb") as stream:
            np.savez_compressed(stream, **arrays)
        temporary.replace(path)
        self.chunks.append(
            {
                "path": path.as_posix(),
                "rows": len(self.records),
                "bytes": path.stat().st_size,
                "sha256": self._sha256(path),
            }
        )
        self.records.clear()
