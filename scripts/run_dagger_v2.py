from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import subprocess
import sys
from dataclasses import fields
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

from riskaware_saferrl.baselines import CausalObservationExpert
from riskaware_saferrl.envs import GridEnvironmentConfig, ResearchConstructionEnvV2
from riskaware_saferrl.policies import RecurrentMaskedPolicy

WORLDS = ("site_small", "site_medium", "site_dynamic")
PROFILES = {
    "low": (0.75, 0.75, 0.0),
    "medium": (1.0, 1.0, 0.1),
    "high": (1.5, 1.5, 0.2),
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_config(world: str, profile: str) -> GridEnvironmentConfig:
    hazard_multiplier, worker_multiplier, noise = PROFILES[profile]
    payload = yaml.safe_load(Path(f"configs/grid/{world}.yaml").read_text(encoding="utf-8"))
    allowed = {field.name for field in fields(GridEnvironmentConfig)}
    payload = {key: value for key, value in payload.items() if key in allowed}
    payload["hazard_density"] = min(0.5, payload["hazard_density"] * hazard_multiplier)
    payload["worker_density"] = min(0.5, payload["worker_density"] * worker_multiplier)
    payload["perception_false_negative_rate"] = noise
    return GridEnvironmentConfig(**payload)


def load_policy(checkpoint_path: Path, device: torch.device) -> RecurrentMaskedPolicy:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    policy = RecurrentMaskedPolicy().to(device)
    policy.load_state_dict(checkpoint["model_state_dict"])
    policy.eval()
    return policy


@torch.inference_mode()
def policy_action(
    policy: RecurrentMaskedPolicy,
    observation: dict[str, np.ndarray],
    hidden: torch.Tensor | None,
    device: torch.device,
) -> tuple[int, torch.Tensor]:
    maps = torch.as_tensor(observation["map"], device=device).unsqueeze(0).unsqueeze(0)
    states = torch.as_tensor(observation["state"], device=device).unsqueeze(0).unsqueeze(0)
    masks = (
        torch.as_tensor(observation["action_mask"], device=device, dtype=torch.bool)
        .unsqueeze(0)
        .unsqueeze(0)
    )
    action, output = policy.predict(maps, states, masks, hidden, deterministic=True)
    return int(action.item()), output.recurrent_state


class CorrectionWriter:
    def __init__(self, directory: Path, chunk_size: int) -> None:
        self.directory = directory
        self.chunk_size = chunk_size
        self.records: list[dict[str, Any]] = []
        self.chunks: list[dict[str, Any]] = []
        self.action_counts = np.zeros(5, dtype=np.int64)
        directory.mkdir(parents=True, exist_ok=True)

    def add(self, record: dict[str, Any]) -> None:
        self.records.append(record)
        self.action_counts[record["expert_action"]] += 1
        if len(self.records) >= self.chunk_size:
            self.flush()

    def flush(self) -> None:
        if not self.records:
            return
        path = self.directory / f"dagger_corrections_{len(self.chunks):05d}.npz"
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite immutable DAgger chunk: {path}")
        arrays = {
            "maps": np.stack([row["map"] for row in self.records]).astype(np.uint8),
            "states": np.stack([row["state"] for row in self.records]).astype(np.float16),
            "action_masks": np.stack([row["action_mask"] for row in self.records]).astype(np.uint8),
            "causal_expert_executed_actions": np.asarray(
                [row["expert_action"] for row in self.records], dtype=np.int8
            ),
            "episode_ids": np.asarray([row["episode_id"] for row in self.records], dtype="U128"),
            "splits": np.asarray(["train"] * len(self.records), dtype="U10"),
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
                "sha256": sha256_file(path),
            }
        )
        self.records.clear()


def collect_iteration(
    *,
    policy: RecurrentMaskedPolicy,
    device: torch.device,
    iteration: int,
    beta: float,
    target_transitions: int,
    seed: int,
    output_dir: Path,
    chunk_size: int,
    expert_strategy: str = "systematic",
) -> dict[str, Any]:
    rng = random.Random(seed)
    writer = CorrectionWriter(output_dir / "chunks", chunk_size)
    transitions = disagreements = invalid = corrected_deadlocks = 0
    episodes: list[dict[str, Any]] = []
    episode_index = 0
    while transitions < target_transitions:
        world = WORLDS[episode_index % len(WORLDS)]
        profile = tuple(PROFILES)[(episode_index // len(WORLDS)) % len(PROFILES)]
        environment = ResearchConstructionEnvV2(load_config(world, profile))
        environment_seed = seed + episode_index
        observation, _ = environment.reset(seed=environment_seed)
        expert = CausalObservationExpert(planning_strategy=expert_strategy)
        hidden = None
        episode_id = f"dagger:{iteration}:{world}:{profile}:{environment_seed}"
        episode_disagreements = 0
        recent_positions: list[tuple[int, int]] = []
        while transitions < target_transitions:
            proposed, hidden = policy_action(policy, observation, hidden, device)
            decision = expert.decide(observation)
            correction = int(decision.final_action)
            if not bool(observation["action_mask"][proposed]):
                invalid += 1
                raise RuntimeError("Masked recurrent policy emitted an invalid action")
            if not bool(observation["action_mask"][correction]):
                invalid += 1
                raise RuntimeError("Causal expert emitted an invalid correction")
            disagreement = proposed != correction
            disagreements += int(disagreement)
            episode_disagreements += int(disagreement)
            position_array = np.argwhere(observation["map"][5] > 0)
            position = tuple(int(value) for value in position_array[0])
            recent_positions.append(position)
            recent_positions = recent_positions[-8:]
            was_deadlocked = len(recent_positions) == 8 and len(set(recent_positions)) <= 2
            executed = correction if rng.random() < beta else proposed
            writer.add(
                {
                    "map": observation["map"],
                    "state": observation["state"],
                    "action_mask": observation["action_mask"],
                    "expert_action": correction,
                    "episode_id": episode_id,
                }
            )
            next_observation, _, terminated, truncated, info = environment.step(executed)
            if was_deadlocked and correction != proposed:
                next_position = tuple(
                    int(value) for value in np.argwhere(next_observation["map"][5] > 0)[0]
                )
                corrected_deadlocks += int(next_position != position or correction == 4)
            observation = next_observation
            transitions += 1
            if terminated or truncated:
                break
        episodes.append(
            {
                "episode_id": episode_id,
                "world": world,
                "profile": profile,
                "seed": environment_seed,
                "success": bool(info["success"]),
                "hazard_recall": float(info["hazard_recall"]),
                "coverage": float(info["inspection_coverage"]),
                "safety_cost": float(environment.cumulative_cost),
                "collisions": int(environment.collisions),
                "steps": int(environment.steps),
                "disagreements": episode_disagreements,
            }
        )
        episode_index += 1
    writer.flush()
    return {
        "iteration": iteration,
        "beta": beta,
        "expert_strategy": expert_strategy,
        "new_visited_state_transitions": transitions,
        "episodes": len(episodes),
        "disagreement_count": disagreements,
        "disagreement_rate": disagreements / max(1, transitions),
        "corrected_deadlocks": corrected_deadlocks,
        "invalid_actions": invalid,
        "action_counts": writer.action_counts.tolist(),
        "chunks": writer.chunks,
        "collection_success_rate": float(np.mean([row["success"] for row in episodes])),
        "collection_hazard_recall": float(np.mean([row["hazard_recall"] for row in episodes])),
        "collection_coverage": float(np.mean([row["coverage"] for row in episodes])),
        "collection_safety_cost": float(np.mean([row["safety_cost"] for row in episodes])),
        "collection_collision_rate": sum(row["collisions"] for row in episodes)
        / max(1, sum(row["steps"] for row in episodes)),
    }


def cumulative_manifest(
    base_manifest_path: Path,
    previous_chunks: list[dict[str, Any]],
    current_chunks: list[dict[str, Any]],
    output_dir: Path,
) -> Path:
    base = json.loads(base_manifest_path.read_text(encoding="utf-8"))
    chunks = [*base["chunks"], *previous_chunks, *current_chunks]
    transition_count = sum(int(chunk["rows"]) for chunk in chunks)
    manifest = {
        **base,
        "status": "PASSED",
        "source": "causal demonstrations plus genuine policy-visited DAgger corrections",
        "transitions": transition_count,
        "chunks": chunks,
        "dataset_sha256": hashlib.sha256(
            "".join(chunk["sha256"] for chunk in chunks).encode()
        ).hexdigest(),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "manifest.json"
    path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return path


def evaluate_policy(
    policy: RecurrentMaskedPolicy,
    device: torch.device,
    seeds: range,
) -> dict[str, Any]:
    episodes: list[dict[str, Any]] = []
    invalid = 0
    for seed in seeds:
        for world in WORLDS:
            for profile in PROFILES:
                environment = ResearchConstructionEnvV2(load_config(world, profile))
                observation, _ = environment.reset(seed=seed)
                hidden = None
                while True:
                    action, hidden = policy_action(policy, observation, hidden, device)
                    invalid += int(not bool(observation["action_mask"][action]))
                    observation, _, terminated, truncated, info = environment.step(action)
                    if terminated or truncated:
                        break
                episodes.append(
                    {
                        "world": world,
                        "profile": profile,
                        "seed": seed,
                        "success": bool(info["success"]),
                        "hazard_recall": float(info["hazard_recall"]),
                        "coverage": float(info["inspection_coverage"]),
                        "safety_cost": float(environment.cumulative_cost),
                        "collisions": int(environment.collisions),
                        "constraint_violations": int(
                            environment.collisions
                            + environment.near_misses
                            + environment.restricted_violations
                        ),
                        "steps": int(environment.steps),
                    }
                )
    world_success = {
        world: float(np.mean([row["success"] for row in episodes if row["world"] == world]))
        for world in WORLDS
    }
    return {
        "episodes": len(episodes),
        "held_out_seeds": list(seeds),
        "success_rate": float(np.mean([row["success"] for row in episodes])),
        "world_success": world_success,
        "hazard_recall": float(np.mean([row["hazard_recall"] for row in episodes])),
        "coverage": float(np.mean([row["coverage"] for row in episodes])),
        "mean_safety_cost": float(np.mean([row["safety_cost"] for row in episodes])),
        "collision_rate": sum(row["collisions"] for row in episodes)
        / max(1, sum(row["steps"] for row in episodes)),
        "constraint_violations": sum(row["constraint_violations"] for row in episodes),
        "invalid_actions": invalid,
    }


def train_iteration(
    *,
    dataset_dir: Path,
    initial_checkpoint: Path,
    output_dir: Path,
    report_dir: Path,
    epochs: int,
    batch_size: int,
    device: str,
    seed: int,
) -> Path:
    command = [
        sys.executable,
        "scripts/train_behavior_cloning_v2.py",
        "--dataset-dir",
        str(dataset_dir),
        "--output-dir",
        str(output_dir),
        "--report-dir",
        str(report_dir),
        "--initial-checkpoint",
        str(initial_checkpoint),
        "--epochs",
        str(epochs),
        "--patience",
        str(max(2, epochs)),
        "--batch-size",
        str(batch_size),
        "--sequence-length",
        "32",
        "--num-workers",
        "0",
        "--max-memory-gb",
        "8",
        "--seed",
        str(seed),
        "--device",
        device,
        "--recurrent-training-state",
    ]
    completed = subprocess.run(command, check=False)
    checkpoint = output_dir / "best_behavior_cloning.pt"
    if not checkpoint.is_file():
        raise RuntimeError(
            f"DAgger retraining exited {completed.returncode} and did not create {checkpoint}"
        )
    return checkpoint


def write_summary_csv(path: Path, iterations: list[dict[str, Any]]) -> None:
    fields_to_write = (
        "iteration",
        "beta",
        "new_visited_state_transitions",
        "disagreement_rate",
        "corrected_deadlocks",
        "held_out_success",
        "held_out_hazard_recall",
        "held_out_coverage",
        "held_out_safety_cost",
        "held_out_collision_rate",
        "invalid_actions",
    )
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields_to_write)
        writer.writeheader()
        writer.writerows({key: row[key] for key in fields_to_write} for row in iterations)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base-dataset",
        type=Path,
        default=Path("artifacts/strong_policy_upgrade/causal_expert_demonstrations_systematic"),
    )
    parser.add_argument("--initial-checkpoint", type=Path, required=True)
    parser.add_argument(
        "--artifact-dir",
        type=Path,
        default=Path("artifacts/strong_policy_upgrade/dagger"),
    )
    parser.add_argument(
        "--report-dir",
        type=Path,
        default=Path("reports/strong_policy_upgrade/dagger"),
    )
    parser.add_argument("--transitions-per-iteration", type=int, default=12_000)
    parser.add_argument("--chunk-size", type=int, default=4_000)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--held-out-seeds", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20260730)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument(
        "--expert-strategy",
        choices=("systematic", "risk_astar"),
        default="systematic",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    args.report_dir.mkdir(parents=True, exist_ok=True)
    current_checkpoint = args.initial_checkpoint
    previous_correction_chunks: list[dict[str, Any]] = []
    results: list[dict[str, Any]] = []
    for iteration, beta in enumerate((0.75, 0.50, 0.25), start=1):
        iteration_artifacts = args.artifact_dir / f"iteration_{iteration}"
        if iteration_artifacts.exists():
            raise FileExistsError(
                f"Refusing to overwrite existing DAgger iteration: {iteration_artifacts}"
            )
        policy = load_policy(current_checkpoint, device)
        collection = collect_iteration(
            policy=policy,
            device=device,
            iteration=iteration,
            beta=beta,
            target_transitions=args.transitions_per_iteration,
            seed=args.seed + iteration * 10_000,
            output_dir=iteration_artifacts,
            chunk_size=args.chunk_size,
            expert_strategy=args.expert_strategy,
        )
        dataset_dir = iteration_artifacts / "cumulative_dataset"
        manifest_path = cumulative_manifest(
            args.base_dataset / "manifest.json",
            previous_correction_chunks,
            collection["chunks"],
            dataset_dir,
        )
        current_checkpoint = train_iteration(
            dataset_dir=dataset_dir,
            initial_checkpoint=current_checkpoint,
            output_dir=iteration_artifacts / "training",
            report_dir=args.report_dir / f"iteration_{iteration}_training",
            epochs=args.epochs,
            batch_size=args.batch_size,
            device=args.device,
            seed=args.seed + iteration,
        )
        trained_policy = load_policy(current_checkpoint, device)
        evaluation = evaluate_policy(
            trained_policy,
            device,
            range(3000, 3000 + args.held_out_seeds),
        )
        result = {
            **collection,
            "cumulative_dataset_manifest": manifest_path.as_posix(),
            "cumulative_transitions": json.loads(manifest_path.read_text(encoding="utf-8"))[
                "transitions"
            ],
            "checkpoint_path": current_checkpoint.as_posix(),
            "checkpoint_sha256": sha256_file(current_checkpoint),
            "held_out_success": evaluation["success_rate"],
            "held_out_hazard_recall": evaluation["hazard_recall"],
            "held_out_coverage": evaluation["coverage"],
            "held_out_safety_cost": evaluation["mean_safety_cost"],
            "held_out_collision_rate": evaluation["collision_rate"],
            "held_out": evaluation,
            "invalid_actions": collection["invalid_actions"] + evaluation["invalid_actions"],
        }
        (args.report_dir / f"dagger_iteration_{iteration}.json").write_text(
            json.dumps(result, indent=2) + "\n", encoding="utf-8"
        )
        results.append(result)
        previous_correction_chunks.extend(collection["chunks"])
    final_policy = load_policy(current_checkpoint, device)
    final_evaluation = evaluate_policy(
        final_policy,
        device,
        range(4000, 4000 + args.held_out_seeds),
    )
    status = "PASSED" if all(row["invalid_actions"] == 0 for row in results) else "FAILED"
    summary = {
        "status": status,
        "genuine_policy_visited_states": True,
        "expert_strategy": args.expert_strategy,
        "expert_mixing_schedule": [0.75, 0.50, 0.25],
        "final_evaluation_beta": 0.0,
        "iterations": results,
        "final_checkpoint": current_checkpoint.as_posix(),
        "final_checkpoint_sha256": sha256_file(current_checkpoint),
        "final_evaluation": final_evaluation,
    }
    (args.report_dir / "dagger_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    write_summary_csv(args.report_dir / "dagger_results.csv", results)
    print(f"DAGGER={status}")
    if status != "PASSED":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
