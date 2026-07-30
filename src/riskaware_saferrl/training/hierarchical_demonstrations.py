from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from riskaware_saferrl.hierarchical.schemas import MissionOption


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True)
class HierarchicalDerivationConfig:
    decision_interval: int = 4
    records_per_chunk: int = 10_000
    context_steps: int = 4


def _robot_position(semantic_map: np.ndarray) -> tuple[int, int]:
    locations = np.argwhere(semantic_map[5] > 0)
    if len(locations) != 1:
        raise ValueError("Every source observation must contain one robot cell")
    return int(locations[0, 0]), int(locations[0, 1])


def _observed_target(semantic_map: np.ndarray) -> tuple[int, int] | None:
    locations = np.argwhere((semantic_map[8] > 0) | (semantic_map[1] > 0))
    if len(locations) == 0:
        return None
    robot = _robot_position(semantic_map)
    ordered = sorted(
        ((int(row), int(column)) for row, column in locations),
        key=lambda value: (
            abs(value[0] - robot[0]) + abs(value[1] - robot[1]),
            value,
        ),
    )
    return ordered[0]


def _frontier_target(semantic_map: np.ndarray) -> tuple[int, int] | None:
    visible = semantic_map[9] > 0
    obstacle = semantic_map[0] > 0
    robot = _robot_position(semantic_map)
    candidates: list[tuple[int, int]] = []
    for row, column in np.argwhere(visible & ~obstacle):
        position = int(row), int(column)
        if any(
            0 <= position[0] + dr < visible.shape[0]
            and 0 <= position[1] + dc < visible.shape[1]
            and not visible[position[0] + dr, position[1] + dc]
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1))
        ):
            candidates.append(position)
    if not candidates:
        return None
    return min(
        candidates,
        key=lambda value: (
            abs(value[0] - robot[0]) + abs(value[1] - robot[1]),
            value,
        ),
    )


def derive_option(
    semantic_map: np.ndarray,
    action_mask: np.ndarray,
    action: int,
    *,
    previous_action: int,
    shield_intervened: bool,
) -> tuple[MissionOption, tuple[int, int] | None, str]:
    """Derive an observation-consistent mission label from causal trajectory structure."""
    robot = _robot_position(semantic_map)
    target = _observed_target(semantic_map)
    worker_distance = min(
        (
            abs(robot[0] - int(row)) + abs(robot[1] - int(column))
            for row, column in np.argwhere((semantic_map[2] > 0) | (semantic_map[7] > 0))
        ),
        default=99,
    )
    local_uncertainty = 1.0 - float(np.mean(semantic_map[9] > 0))
    if action == 4 and target is not None:
        option = (
            MissionOption.INSPECT_PPE_VIOLATION
            if semantic_map[8, target[0], target[1]] > 0
            else MissionOption.INSPECT_KNOWN_RISK
        )
        return option, target, "observed_target_in_inspection_range"
    if worker_distance <= 2:
        return MissionOption.AVOID_DYNAMIC_WORKER, None, "observed_worker_clearance"
    if shield_intervened:
        return MissionOption.REPLAN_ROUTE, target, "observed_shield_intervention"
    if target is not None:
        return (
            (
                MissionOption.INSPECT_PPE_VIOLATION
                if semantic_map[8, target[0], target[1]] > 0
                else MissionOption.INSPECT_KNOWN_RISK
            ),
            target,
            "observed_or_remembered_target",
        )
    if local_uncertainty > 0.9 and action == 4 and bool(action_mask[4]):
        return MissionOption.HOLD_FOR_UNCERTAINTY, None, "high_observation_uncertainty"
    return MissionOption.EXPLORE_FRONTIER, _frontier_target(semantic_map), "observed_frontier"


def _vector_cost(semantic_map: np.ndarray, safety_cost: float) -> np.ndarray:
    robot = _robot_position(semantic_map)
    restricted = float(semantic_map[3, robot[0], robot[1]] > 0)
    worker = float(
        any(
            abs(robot[0] - int(row)) + abs(robot[1] - int(column)) < 1
            for row, column in np.argwhere(semantic_map[2] > 0)
        )
    )
    semantic = float(max(semantic_map[6, robot[0], robot[1]], semantic_map[8, robot[0], robot[1]]))
    uncertainty = 1.0 - float(np.mean(semantic_map[9] > 0))
    return np.asarray(
        (0.0, restricted, worker, semantic, uncertainty, 0.0, 0.0),
        dtype=np.float32,
    ) * max(1.0, float(safety_cost))


class HierarchicalDatasetBuilder:
    """Streaming derivation that never mutates or fully loads the source dataset."""

    def __init__(
        self,
        source_dir: Path,
        output_dir: Path,
        config: HierarchicalDerivationConfig | None = None,
    ) -> None:
        self.source_dir = source_dir
        self.output_dir = output_dir
        self.config = config or HierarchicalDerivationConfig()

    def _source_manifest(self) -> dict[str, Any]:
        return json.loads((self.source_dir / "manifest.json").read_text(encoding="utf-8"))

    @staticmethod
    def _empty_records() -> dict[str, list[Any]]:
        return {
            "maps": [],
            "states": [],
            "option_masks": [],
            "options": [],
            "target_coordinates": [],
            "target_types": [],
            "option_durations": [],
            "planner_requests": [],
            "planner_outcomes": [],
            "executed_primitives": [],
            "executed_primitive_masks": [],
            "shield_interventions": [],
            "mission_progress": [],
            "rewards": [],
            "vector_costs": [],
            "terminated": [],
            "truncated": [],
            "success": [],
            "episode_ids": [],
            "episode_steps": [],
            "splits": [],
            "recurrent_context": [],
        }

    def _write_chunk(self, records: dict[str, list[Any]], index: int) -> dict[str, Any]:
        chunk_dir = self.output_dir / "chunks"
        chunk_dir.mkdir(parents=True, exist_ok=True)
        path = chunk_dir / f"hierarchical_demonstrations_{index:05d}.npz"
        arrays = {key: np.asarray(value) for key, value in records.items()}
        np.savez_compressed(path, **arrays)
        return {
            "path": path.as_posix(),
            "rows": len(records["options"]),
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }

    def build(self) -> dict[str, Any]:
        if self.output_dir.exists() and (self.output_dir / "manifest.json").exists():
            existing = json.loads((self.output_dir / "manifest.json").read_text(encoding="utf-8"))
            if existing.get("status") == "PASSED":
                return existing
            raise RuntimeError("Incomplete hierarchical output requires explicit repair")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        source = self._source_manifest()
        source_hashes_before = {
            item["path"]: sha256_file(Path(item["path"])) for item in source["chunks"]
        }
        chunks: list[dict[str, Any]] = []
        records = self._empty_records()
        split_counts: dict[str, int] = {}
        episode_splits: dict[str, str] = {}
        option_counts = np.zeros(len(MissionOption), dtype=np.int64)
        for source_chunk in source["chunks"]:
            with np.load(Path(source_chunk["path"]), allow_pickle=False) as data:
                required_keys = (
                    "maps",
                    "states",
                    "action_masks",
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
                )
                chunk_data = {key: data[key] for key in required_keys}
                rows = len(chunk_data["episode_ids"])
                for index in range(rows):
                    step = int(chunk_data["episode_steps"][index])
                    if step % self.config.decision_interval != 0:
                        continue
                    episode_id = str(chunk_data["episode_ids"][index])
                    split = str(chunk_data["splits"][index])
                    previous_split = episode_splits.setdefault(episode_id, split)
                    if previous_split != split:
                        raise ValueError("Episode split leakage detected in source")
                    semantic_map = np.asarray(chunk_data["maps"][index])
                    action = int(chunk_data["causal_expert_executed_actions"][index])
                    previous_action = (
                        int(chunk_data["causal_expert_executed_actions"][index - 1])
                        if index > 0 and str(chunk_data["episode_ids"][index - 1]) == episode_id
                        else 4
                    )
                    shield_value = str(chunk_data["shield_decisions"][index])
                    option, target, reason = derive_option(
                        semantic_map,
                        np.asarray(chunk_data["action_masks"][index]),
                        action,
                        previous_action=previous_action,
                        shield_intervened=shield_value != "accept",
                    )
                    primitives = np.full(self.config.decision_interval, 4, dtype=np.int8)
                    primitive_mask = np.zeros(self.config.decision_interval, dtype=np.bool_)
                    duration = 0
                    for offset in range(self.config.decision_interval):
                        candidate = index + offset
                        if (
                            candidate >= rows
                            or str(chunk_data["episode_ids"][candidate]) != episode_id
                        ):
                            break
                        primitives[offset] = chunk_data["causal_expert_executed_actions"][candidate]
                        primitive_mask[offset] = True
                        duration += 1
                    context = np.zeros((self.config.context_steps, 3), dtype=np.float32)
                    for context_offset in range(self.config.context_steps):
                        candidate = index - self.config.context_steps + context_offset
                        if candidate < 0 or str(chunk_data["episode_ids"][candidate]) != episode_id:
                            continue
                        context[context_offset] = (
                            float(chunk_data["causal_expert_executed_actions"][candidate]),
                            float(chunk_data["rewards"][candidate]),
                            float(chunk_data["safety_costs"][candidate]),
                        )
                    records["maps"].append(semantic_map)
                    records["states"].append(chunk_data["states"][index])
                    records["option_masks"].append(np.ones(len(MissionOption), dtype=np.uint8))
                    records["options"].append(int(option))
                    records["target_coordinates"].append(target if target is not None else (-1, -1))
                    records["target_types"].append(reason)
                    records["option_durations"].append(duration)
                    records["planner_requests"].append(option.name.lower())
                    records["planner_outcomes"].append("source_action_valid")
                    records["executed_primitives"].append(primitives)
                    records["executed_primitive_masks"].append(primitive_mask)
                    records["shield_interventions"].append(shield_value != "accept")
                    records["mission_progress"].append(chunk_data["mission_progress"][index])
                    records["rewards"].append(chunk_data["rewards"][index])
                    records["vector_costs"].append(
                        _vector_cost(
                            semantic_map,
                            float(chunk_data["safety_costs"][index]),
                        )
                    )
                    records["terminated"].append(chunk_data["terminated"][index])
                    records["truncated"].append(chunk_data["truncated"][index])
                    records["success"].append(chunk_data["success"][index])
                    records["episode_ids"].append(episode_id)
                    records["episode_steps"].append(step)
                    records["splits"].append(split)
                    records["recurrent_context"].append(context)
                    split_counts[split] = split_counts.get(split, 0) + 1
                    option_counts[int(option)] += 1
                    if len(records["options"]) >= self.config.records_per_chunk:
                        chunks.append(self._write_chunk(records, len(chunks)))
                        records = self._empty_records()
        if records["options"]:
            chunks.append(self._write_chunk(records, len(chunks)))
        source_hashes_after = {
            item["path"]: sha256_file(Path(item["path"])) for item in source["chunks"]
        }
        if source_hashes_before != source_hashes_after:
            raise RuntimeError("Immutable causal source changed during derivation")
        dataset_digest = hashlib.sha256(
            "".join(item["sha256"] for item in chunks).encode()
        ).hexdigest()
        manifest = {
            "status": "PASSED",
            "observation_only_derivation": True,
            "hidden_state_consumed": False,
            "source_manifest": (self.source_dir / "manifest.json").as_posix(),
            "source_dataset_sha256": source["dataset_sha256"],
            "source_chunk_hashes_unchanged": True,
            "decision_interval": self.config.decision_interval,
            "transitions": int(sum(item["rows"] for item in chunks)),
            "episodes": len(episode_splits),
            "split_counts": split_counts,
            "episode_split_integrity": True,
            "option_counts": option_counts.tolist(),
            "chunks": chunks,
            "dataset_sha256": dataset_digest,
        }
        (self.output_dir / "manifest.json").write_text(
            json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
        )
        return manifest
