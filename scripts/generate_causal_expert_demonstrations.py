from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import fields
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from riskaware_saferrl.baselines import CausalObservationExpert
from riskaware_saferrl.envs import GridEnvironmentConfig, ResearchConstructionEnvV2

SCENARIOS = ("site_small", "site_medium", "site_dynamic")
HAZARD_MULTIPLIERS = (0.75, 1.0, 1.5)
WORKER_MULTIPLIERS = (0.75, 1.0, 1.5)
NOISE_LEVELS = (0.0, 0.1, 0.2)
ACTION_COUNT = 5


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_config(
    name: str,
    hazard_multiplier: float,
    worker_multiplier: float,
    noise: float,
) -> GridEnvironmentConfig:
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


class CausalChunkWriter:
    def __init__(self, root: Path, chunk_size: int) -> None:
        self.root = root
        self.chunk_size = chunk_size
        self.root.mkdir(parents=True, exist_ok=True)
        self.records: list[dict[str, Any]] = []
        self.chunks: list[dict[str, Any]] = []
        self.total = 0
        self.action_counts = np.zeros(ACTION_COUNT, dtype=np.int64)

    def add(self, record: dict[str, Any]) -> None:
        self.records.append(record)
        self.action_counts[int(record["causal_expert_executed_action"])] += 1
        if len(self.records) >= self.chunk_size:
            self.flush()

    def flush(self) -> None:
        if not self.records:
            return
        path = self.root / f"causal_demonstrations_{len(self.chunks):05d}.npz"
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
            "causal_expert_proposed_actions": np.asarray(
                [r["causal_expert_proposed_action"] for r in self.records],
                dtype=np.int8,
            ),
            "causal_expert_executed_actions": np.asarray(
                [r["causal_expert_executed_action"] for r in self.records],
                dtype=np.int8,
            ),
            "shield_decisions": np.asarray(
                [r["shield_decision"] for r in self.records], dtype="U7"
            ),
            "rewards": np.asarray([r["reward"] for r in self.records], dtype=np.float32),
            "safety_costs": np.asarray([r["safety_cost"] for r in self.records], dtype=np.float32),
            "terminated": np.asarray([r["terminated"] for r in self.records], dtype=np.bool_),
            "truncated": np.asarray([r["truncated"] for r in self.records], dtype=np.bool_),
            "success": np.asarray([r["success"] for r in self.records], dtype=np.bool_),
            "mission_progress": np.asarray(
                [r["mission_progress"] for r in self.records], dtype=np.float32
            ),
            "episode_ids": np.asarray([r["episode_id"] for r in self.records], dtype="U128"),
            "episode_steps": np.asarray([r["episode_step"] for r in self.records], dtype=np.int16),
            "splits": np.asarray([r["split"] for r in self.records], dtype="U10"),
            "discovered_target_counts": np.asarray(
                [r["discovered_target_count"] for r in self.records],
                dtype=np.int16,
            ),
            "target_types": np.asarray([r["target_type"] for r in self.records], dtype="U20"),
        }
        temporary = path.with_suffix(".npz.tmp")
        with temporary.open("wb") as stream:
            np.savez_compressed(stream, **arrays)
        temporary.replace(path)
        rows = len(self.records)
        self.chunks.append(
            {
                "path": path.as_posix(),
                "rows": rows,
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
        self.total += rows
        self.records.clear()


def _split_counts(chunks: list[dict[str, Any]]) -> dict[str, int]:
    counts = {"train": 0, "validation": 0, "test": 0}
    for chunk in chunks:
        with np.load(chunk["path"], allow_pickle=False) as payload:
            for split in counts:
                counts[split] += int(np.count_nonzero(payload["splits"] == split))
    return counts


def generate(
    output_dir: Path,
    *,
    target_transitions: int,
    chunk_size: int,
    seeds: int,
    minimum_unique_seeds: int,
    planning_strategy: str = "systematic",
) -> dict[str, Any]:
    if (
        output_dir.resolve()
        == Path("artifacts/strong_policy_upgrade/expert_demonstrations").resolve()
    ):
        raise ValueError("Causal demonstrations must not overwrite the original dataset")
    if any((output_dir / "chunks").glob("*.npz")):
        raise FileExistsError(f"Refusing to overwrite immutable causal chunks in {output_dir}")
    writer = CausalChunkWriter(output_dir / "chunks", chunk_size)
    conditions = [
        (scenario, hazard, worker, noise)
        for scenario in SCENARIOS
        for hazard in HAZARD_MULTIPLIERS
        for worker in WORKER_MULTIPLIERS
        for noise in NOISE_LEVELS
    ]
    episodes: list[dict[str, Any]] = []
    used_seeds: set[int] = set()
    rejected_transitions = 0
    invalid_actions = 0
    shield_interventions = 0
    attempted = 0
    maximum_episodes = max(conditions.__len__() * seeds * 4, seeds)
    for episode_index in range(maximum_episodes):
        seed = episode_index % seeds
        used_seeds.add(seed)
        scenario, hazard_multiplier, worker_multiplier, noise = conditions[
            episode_index % len(conditions)
        ]
        environment = ResearchConstructionEnvV2(
            load_config(
                scenario,
                hazard_multiplier,
                worker_multiplier,
                noise,
            )
        )
        observation, _ = environment.reset(seed=seed)
        expert = CausalObservationExpert(planning_strategy=planning_strategy)
        episode_id = (
            f"causal:{scenario}:h{hazard_multiplier}:w{worker_multiplier}:"
            f"n{noise}:s{seed}:e{episode_index}"
        )
        split = split_for_seed(seed)
        records: list[dict[str, Any]] = []
        episode_step = 0
        while True:
            decision = expert.decide(observation)
            executed = int(decision.final_action)
            is_valid = bool(observation["action_mask"][executed])
            invalid_actions += int(not is_valid)
            if not is_valid:
                raise RuntimeError("Causal expert emitted an invalid action")
            shield_interventions += int(decision.shield_decision != "accept")
            discovered_target_count = expert.discovered_target_count
            next_observation, reward, terminated, truncated, info = environment.step(executed)
            records.append(
                {
                    "observation": observation,
                    "next_observation": next_observation,
                    "causal_expert_proposed_action": decision.proposed_action,
                    "causal_expert_executed_action": executed,
                    "shield_decision": decision.shield_decision,
                    "reward": reward,
                    "safety_cost": info["cost"],
                    "terminated": terminated,
                    "truncated": truncated,
                    "success": bool(info["success"]),
                    "mission_progress": info["inspection_coverage"],
                    "episode_id": episode_id,
                    "episode_step": episode_step,
                    "split": split,
                    "discovered_target_count": discovered_target_count,
                    "target_type": decision.target_type,
                }
            )
            observation = next_observation
            episode_step += 1
            if terminated or truncated:
                break
        success = bool(info["success"])
        if success:
            for record in records:
                writer.add(record)
        else:
            rejected_transitions += len(records)
        episodes.append(
            {
                "episode_id": episode_id,
                "seed": seed,
                "split": split,
                "scenario": scenario,
                "hazard_multiplier": hazard_multiplier,
                "worker_multiplier": worker_multiplier,
                "perception_false_negative_rate": noise,
                "steps": episode_step,
                "success": success,
                "hazard_recall": float(info["hazard_recall"]),
                "safety_cost": float(environment.cumulative_cost),
            }
        )
        attempted += 1
        if (
            writer.total + len(writer.records) >= target_transitions
            and len(used_seeds) >= minimum_unique_seeds
        ):
            break
    writer.flush()
    split_counts = _split_counts(writer.chunks)
    dataset_hash = hashlib.sha256(
        "".join(chunk["sha256"] for chunk in writer.chunks).encode()
    ).hexdigest()
    episode_split_integrity = all(
        len({row["split"] for row in episodes if row["episode_id"] == episode_id}) == 1
        for episode_id in {row["episode_id"] for row in episodes}
    )
    passed = (
        writer.total >= target_transitions
        and len(used_seeds) >= minimum_unique_seeds
        and invalid_actions == 0
        and episode_split_integrity
    )
    return {
        "status": "PASSED" if passed else "FAILED",
        "observation_only_contract": True,
        "planning_strategy": planning_strategy,
        "policy_observation_keys": ["map", "state", "action_mask"],
        "hidden_state_stored_or_consumed": False,
        "transitions": writer.total,
        "target_transitions": target_transitions,
        "episodes": attempted,
        "successful_episodes": sum(row["success"] for row in episodes),
        "success_rate": float(np.mean([row["success"] for row in episodes])),
        "unique_seeds": len(used_seeds),
        "minimum_unique_seeds": minimum_unique_seeds,
        "split_transition_counts": split_counts,
        "split_policy": "episode seed modulo 10 (8=validation, 9=test, others=train)",
        "episode_split_integrity": episode_split_integrity,
        "invalid_actions": invalid_actions,
        "shield_interventions": shield_interventions,
        "failed_episode_transitions_rejected": rejected_transitions,
        "per_action_counts": writer.action_counts.tolist(),
        "action_key": "causal_expert_executed_actions",
        "dataset_sha256": dataset_hash,
        "chunks": writer.chunks,
        "episode_summaries": episodes,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/strong_policy_upgrade/causal_expert_demonstrations"),
    )
    parser.add_argument("--target-transitions", type=int, default=250_000)
    parser.add_argument("--chunk-size", type=int, default=10_000)
    parser.add_argument("--seeds", type=int, default=200)
    parser.add_argument("--minimum-unique-seeds", type=int, default=200)
    parser.add_argument(
        "--planning-strategy",
        choices=("systematic", "risk_astar"),
        default="systematic",
    )
    parser.add_argument(
        "--report-dir",
        type=Path,
        default=Path("reports/strong_policy_upgrade/causal_expert_dataset"),
    )
    args = parser.parse_args()
    summary = generate(
        args.output_dir,
        target_transitions=args.target_transitions,
        chunk_size=args.chunk_size,
        seeds=args.seeds,
        minimum_unique_seeds=args.minimum_unique_seeds,
        planning_strategy=args.planning_strategy,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = dict(summary)
    manifest.pop("episode_summaries")
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    report_dir = args.report_dir
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"CAUSAL_EXPERT_DATASET={summary['status']}")
    if summary["status"] != "PASSED":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
