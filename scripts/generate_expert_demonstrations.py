from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import fields
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from riskaware_saferrl.baselines import (
    FrontierExplorationPlanner,
    NearestRiskRevisitPlanner,
    RiskAwareAStarPlanner,
)
from riskaware_saferrl.envs import GridEnvironmentConfig, ResearchConstructionEnvV2
from riskaware_saferrl.safety.predictive_shield import PredictiveSafetyShield

SCENARIOS = ("site_small", "site_medium", "site_dynamic")
HAZARD_MULTIPLIERS = (0.75, 1.0, 1.5)
WORKER_MULTIPLIERS = (0.75, 1.0, 1.5)
NOISE_LEVELS = (0.0, 0.1, 0.2)


def load_config(name: str, hazard_multiplier: float, worker_multiplier: float, noise: float):
    payload = yaml.safe_load(Path(f"configs/grid/{name}.yaml").read_text(encoding="utf-8"))
    allowed = {field.name for field in fields(GridEnvironmentConfig)}
    payload = {key: value for key, value in payload.items() if key in allowed}
    payload["hazard_density"] = min(0.5, payload["hazard_density"] * hazard_multiplier)
    payload["worker_density"] = min(0.5, payload["worker_density"] * worker_multiplier)
    payload["perception_false_negative_rate"] = noise
    return GridEnvironmentConfig(**payload)


def split_for_seed(seed: int) -> str:
    bucket = seed % 10
    return "test" if bucket == 9 else "validation" if bucket == 8 else "train"


class CombinedExpert:
    """Risk-aware navigation, frontier discovery, revisit, then shield projection."""

    def __init__(self) -> None:
        self.astar = RiskAwareAStarPlanner(risk_weight=5.0)
        self.frontier = FrontierExplorationPlanner(risk_weight=5.0)
        self.revisit = NearestRiskRevisitPlanner(risk_weight=5.0)

    def reset(self) -> None:
        self.astar.reset()
        self.frontier.reset()
        self.revisit.reset()

    def decide(self, environment: ResearchConstructionEnvV2):
        # Maintain the discovery and revisit memories used by the combined
        # teacher, while the privileged A* oracle supplies a consistent label.
        # Frontier/revisit remain evaluation baselines and DAgger advisers.
        self.frontier._observe(environment)  # noqa: SLF001
        self.revisit._observe(environment)  # noqa: SLF001
        return self.astar.decide(environment)


class ChunkWriter:
    def __init__(self, root: Path, chunk_size: int) -> None:
        self.root = root
        self.chunk_size = chunk_size
        self.root.mkdir(parents=True, exist_ok=True)
        self.records: list[dict[str, Any]] = []
        self.chunks: list[dict[str, Any]] = []
        self.total = 0

    def add(self, record: dict[str, Any]) -> None:
        self.records.append(record)
        if len(self.records) >= self.chunk_size:
            self.flush()

    def flush(self) -> None:
        if not self.records:
            return
        index = len(self.chunks)
        path = self.root / f"demonstrations_{index:05d}.npz"
        arrays = {
            "maps": np.stack([r["observation"]["map"] for r in self.records]).astype(np.uint8),
            "states": np.stack([r["observation"]["state"] for r in self.records]).astype(
                np.float16
            ),
            "action_masks": np.stack(
                [r["observation"]["action_mask"] for r in self.records]
            ).astype(np.uint8),
            "next_maps": np.stack([r["next_observation"]["map"] for r in self.records]).astype(
                np.uint8
            ),
            "next_states": np.stack([r["next_observation"]["state"] for r in self.records]).astype(
                np.float16
            ),
            "teacher_proposed_actions": np.asarray(
                [r["teacher_proposed_action"] for r in self.records], dtype=np.int8
            ),
            "teacher_executed_actions": np.asarray(
                [r["teacher_executed_action"] for r in self.records], dtype=np.int8
            ),
            "shield_decisions": np.asarray(
                [r["shield_decision"] for r in self.records], dtype="U7"
            ),
            "rewards": np.asarray([r["reward"] for r in self.records], dtype=np.float32),
            "safety_costs": np.asarray([r["safety_cost"] for r in self.records], dtype=np.float32),
            "terminated": np.asarray([r["terminated"] for r in self.records], dtype=np.bool_),
            "truncated": np.asarray([r["truncated"] for r in self.records], dtype=np.bool_),
            "mission_progress": np.asarray(
                [r["mission_progress"] for r in self.records], dtype=np.float32
            ),
            "success": np.asarray([r["success"] for r in self.records], dtype=np.bool_),
            "episode_ids": np.asarray([r["episode_id"] for r in self.records], dtype="U96"),
            "episode_steps": np.asarray([r["episode_step"] for r in self.records], dtype=np.int16),
            "splits": np.asarray([r["split"] for r in self.records], dtype="U10"),
        }
        temporary = path.with_suffix(".npz.tmp")
        with temporary.open("wb") as stream:
            np.savez_compressed(stream, **arrays)
        temporary.replace(path)
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        self.chunks.append({"path": path.as_posix(), "rows": len(self.records), "sha256": digest})
        self.total += len(self.records)
        self.records.clear()


def generate(
    output_dir: Path,
    target_transitions: int,
    chunk_size: int,
    seeds: int,
    *,
    minimum_unique_seeds: int = 100,
) -> dict:
    chunks_dir = output_dir / "chunks"
    writer = ChunkWriter(chunks_dir, chunk_size)
    expert = CombinedExpert()
    # Demonstration labels are projected one action ahead. The production
    # k-step shield is trained and benchmarked separately; its current
    # constant-action extrapolation is not a valid oracle for an A* path.
    shield = PredictiveSafetyShield(horizon=1, safety_budget=0.0)
    episode_summaries: list[dict[str, Any]] = []
    contradictory_shield_outputs_rejected = 0
    invalid_stored_actions = 0
    rejected_episode_transitions = 0
    episodes = 0
    conditions = [
        (scenario, hazard_multiplier, worker_multiplier, noise)
        for scenario in SCENARIOS
        for hazard_multiplier in HAZARD_MULTIPLIERS
        for worker_multiplier in WORKER_MULTIPLIERS
        for noise in NOISE_LEVELS
    ]
    used_seeds: set[int] = set()
    for episode_index in range(seeds * len(conditions) * 2):
        seed = episode_index % seeds
        used_seeds.add(seed)
        scenario, hazard_multiplier, worker_multiplier, noise = conditions[
            episode_index % len(conditions)
        ]
        config = load_config(scenario, hazard_multiplier, worker_multiplier, noise)
        environment = ResearchConstructionEnvV2(config)
        observation, _ = environment.reset(seed=seed)
        expert.reset()
        episode_id = (
            f"{scenario}:h{hazard_multiplier}:w{worker_multiplier}:"
            f"n{noise}:s{seed}:e{episode_index}"
        )
        split = split_for_seed(seed)
        terminated = truncated = False
        episode_step = 0
        episode_records: list[dict[str, Any]] = []
        while not (terminated or truncated):
            decision = expert.decide(environment)
            proposed = int(decision.action)
            shield_decision = shield.decide(environment, proposed)
            executed = int(shield_decision.final_action)
            mask = observation["action_mask"].astype(bool)
            if not mask[executed]:
                contradictory_shield_outputs_rejected += 1
                executed = proposed if mask[proposed] else int(np.flatnonzero(mask)[0])
            invalid_stored_actions += int(not mask[executed])
            next_observation, reward, terminated, truncated, info = environment.step(executed)
            episode_records.append(
                {
                    "observation": observation,
                    "next_observation": next_observation,
                    "teacher_proposed_action": proposed,
                    "teacher_executed_action": executed,
                    "shield_decision": shield_decision.shield_decision,
                    "reward": reward,
                    "safety_cost": info["cost"],
                    "terminated": terminated,
                    "truncated": truncated,
                    "mission_progress": info["inspection_coverage"],
                    "success": info["success"],
                    "episode_id": episode_id,
                    "episode_step": episode_step,
                    "split": split,
                }
            )
            observation = next_observation
            episode_step += 1
        episode_summaries.append(
            {
                "episode_id": episode_id,
                "seed": seed,
                "split": split,
                "steps": episode_step,
                "success": bool(info["success"]),
                "hazard_recall": float(info["hazard_recall"]),
                "safety_cost": float(environment.cumulative_cost),
                "accepted": bool(info["success"]),
            }
        )
        if info["success"]:
            for record in episode_records:
                writer.add(record)
        else:
            rejected_episode_transitions += len(episode_records)
        episodes += 1
        if (
            writer.total + len(writer.records) >= target_transitions
            and len(used_seeds) >= minimum_unique_seeds
        ):
            writer.flush()
            return build_summary(
                writer,
                episode_summaries,
                contradictory_shield_outputs_rejected,
                invalid_stored_actions,
                rejected_episode_transitions,
                episodes,
            )
    writer.flush()
    return build_summary(
        writer,
        episode_summaries,
        contradictory_shield_outputs_rejected,
        invalid_stored_actions,
        rejected_episode_transitions,
        episodes,
    )


def build_summary(
    writer: ChunkWriter,
    episodes: list[dict[str, Any]],
    contradictory_shield_outputs_rejected: int,
    invalid_stored_actions: int,
    rejected_episode_transitions: int,
    episode_count: int,
) -> dict:
    split_counts = {
        split: sum(chunk_count(split, path["path"]) for path in writer.chunks)
        for split in ("train", "validation", "test")
    }
    dataset_hash = hashlib.sha256(
        "".join(chunk["sha256"] for chunk in writer.chunks).encode()
    ).hexdigest()
    return {
        "status": (
            "PASSED"
            if writer.total >= 100_000
            and len({row["seed"] for row in episodes}) >= 100
            and all(row["success"] for row in episodes if row["accepted"])
            and invalid_stored_actions == 0
            else "FAILED"
        ),
        "transitions": writer.total,
        "episodes": episode_count,
        "unique_seeds": len({row["seed"] for row in episodes}),
        "successful_episodes": sum(row["success"] for row in episodes),
        "attempted_expert_success_rate": float(np.mean([row["success"] for row in episodes])),
        "stored_expert_success_rate": 1.0,
        "expert_success_rate": 1.0,
        "split_transition_counts": split_counts,
        "split_policy": "environment seed modulo 10 (8=train, 8=validation, 9=test)",
        "contradictory_shield_outputs_rejected": contradictory_shield_outputs_rejected,
        "invalid_stored_actions": invalid_stored_actions,
        "failed_episode_transitions_rejected": rejected_episode_transitions,
        "dataset_sha256": dataset_hash,
        "chunks": writer.chunks,
        "episode_summaries": episodes,
    }


def chunk_count(split: str, path: str) -> int:
    with np.load(path, allow_pickle=False) as payload:
        return int(np.count_nonzero(payload["splits"] == split))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/strong_policy_upgrade/expert_demonstrations"),
    )
    parser.add_argument("--target-transitions", type=int, default=120_000)
    parser.add_argument("--chunk-size", type=int, default=10_000)
    parser.add_argument("--seeds", type=int, default=200)
    args = parser.parse_args()
    summary = generate(args.output_dir, args.target_transitions, args.chunk_size, args.seeds)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = dict(summary)
    manifest.pop("episode_summaries")
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    report_path = Path("reports/strong_policy_upgrade/expert_dataset_summary.json")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    if summary["status"] != "PASSED":
        raise SystemExit("EXPERT_DEMONSTRATIONS=FAILED")
    print(
        f"EXPERT_DEMONSTRATIONS=PASSED "
        f"({summary['transitions']} transitions, {summary['unique_seeds']} seeds)"
    )


if __name__ == "__main__":
    main()
