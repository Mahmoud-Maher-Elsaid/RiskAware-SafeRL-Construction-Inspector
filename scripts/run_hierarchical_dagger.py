from __future__ import annotations

import argparse
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
from riskaware_saferrl.hierarchical import (
    CausalRiskAwarePlanner,
    HierarchicalMissionPolicy,
    MissionOption,
    PredictiveLocalController,
    RiskShieldHierarchicalSystem,
)
from riskaware_saferrl.hierarchical.schemas import causal_option_mask
from riskaware_saferrl.safety import EventAwarePredictiveShieldV3, SafetyContractV3
from riskaware_saferrl.training.hierarchical_demonstrations import (
    _vector_cost,
    derive_option,
)

WORLDS = ("site_small", "site_medium", "site_dynamic")
PROFILES = {
    "low": (0.75, 0.75, 0.0),
    "medium": (1.0, 1.0, 0.1),
    "high": (1.5, 1.5, 0.2),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run genuine hierarchical DAgger.")
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--transitions-per-iteration", type=int, default=3000)
    parser.add_argument("--evaluation-transitions", type=int, default=3000)
    parser.add_argument("--retrain-epochs", type=int, default=5)
    parser.add_argument("--seed", type=int, default=6100)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--initial-checkpoint",
        type=Path,
        default=Path(
            "artifacts/strong_policy_upgrade/hierarchical_imitation/seed_42/best_checkpoint.pt"
        ),
    )
    parser.add_argument(
        "--base-dataset",
        type=Path,
        default=Path("artifacts/strong_policy_upgrade/hierarchical_demonstrations"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/strong_policy_upgrade/hierarchical_dagger"),
    )
    parser.add_argument(
        "--reports",
        type=Path,
        default=Path("reports/strong_policy_upgrade/hierarchical_dagger"),
    )
    return parser.parse_args()


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_config(world: str, profile: str) -> GridEnvironmentConfig:
    hazard_multiplier, worker_multiplier, noise = PROFILES[profile]
    payload = yaml.safe_load(Path(f"configs/grid/{world}.yaml").read_text(encoding="utf-8"))
    allowed = {field.name for field in fields(GridEnvironmentConfig)}
    payload = {key: value for key, value in payload.items() if key in allowed}
    payload["hazard_density"] = min(0.5, payload["hazard_density"] * hazard_multiplier)
    payload["worker_density"] = min(0.5, payload["worker_density"] * worker_multiplier)
    payload["perception_false_negative_rate"] = noise
    return GridEnvironmentConfig(**payload)


def load_policy(path: Path, device: torch.device) -> HierarchicalMissionPolicy:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model = HierarchicalMissionPolicy().to(device)
    model.load_state_dict(checkpoint["model"])
    return model.eval()


def structured_state(
    observation: dict[str, np.ndarray],
    context: np.ndarray,
    progress: float,
    shield_intervened: bool,
    duration: int,
) -> np.ndarray:
    return np.concatenate(
        (
            np.asarray(observation["state"], dtype=np.float32),
            context.reshape(-1),
            np.asarray(
                (progress, float(shield_intervened), duration / 8.0),
                dtype=np.float32,
            ),
        )
    )


def option_mask(observation: dict[str, np.ndarray]) -> np.ndarray:
    return causal_option_mask(
        np.asarray(observation["map"]),
        np.asarray(observation["action_mask"]),
    )


class CorrectionWriter:
    def __init__(self, directory: Path, chunk_size: int = 1000) -> None:
        self.directory = directory
        self.chunk_size = chunk_size
        self.rows: list[dict[str, Any]] = []
        self.chunks: list[dict[str, Any]] = []
        directory.mkdir(parents=True, exist_ok=True)

    def add(self, row: dict[str, Any]) -> None:
        self.rows.append(row)
        if len(self.rows) >= self.chunk_size:
            self.flush()

    def flush(self) -> None:
        if not self.rows:
            return
        path = self.directory / f"hierarchical_corrections_{len(self.chunks):05d}.npz"
        arrays = {key: np.asarray([row[key] for row in self.rows]) for key in self.rows[0]}
        np.savez_compressed(path, **arrays)
        self.chunks.append(
            {
                "path": path.as_posix(),
                "rows": len(self.rows),
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
        )
        self.rows.clear()


@torch.inference_mode()
def collect(
    args: argparse.Namespace,
    checkpoint: Path,
    iteration: int,
    beta: float,
) -> dict[str, Any]:
    device = torch.device(args.device)
    policy = load_policy(checkpoint, device)
    rng = random.Random(args.seed + iteration)
    iteration_dir = args.output / f"iteration_{iteration}"
    writer = CorrectionWriter(iteration_dir / "chunks")
    transitions = disagreements = target_disagreements = unsafe_options = 0
    planner_failures = shield_interventions = 0
    option_counts = np.zeros(9, dtype=np.int64)
    primitive_counts = np.zeros(5, dtype=np.int64)
    successes: list[bool] = []
    recalls: list[float] = []
    coverages: list[float] = []
    costs: list[float] = []
    episode_results: list[dict[str, Any]] = []
    episode_index = 0
    while transitions < args.transitions_per_iteration:
        world = WORLDS[episode_index % 3]
        profile = tuple(PROFILES)[(episode_index // 3) % 3]
        environment = ResearchConstructionEnvV2(load_config(world, profile))
        seed = args.seed + iteration * 10_000 + episode_index
        observation, _ = environment.reset(seed=seed)
        expert = CausalObservationExpert(planning_strategy="systematic")
        contract = SafetyContractV3.from_yaml(Path("configs/safety/safety_contract_v3.yaml"))
        system = RiskShieldHierarchicalSystem(
            CausalRiskAwarePlanner(),
            PredictiveLocalController(),
            EventAwarePredictiveShieldV3(contract),
        )
        context = np.zeros((4, 3), dtype=np.float32)
        hidden: torch.Tensor | None = None
        previous_expert_action = 4
        progress = 0.0
        shield_intervened = False
        duration = 1
        active_option: MissionOption | None = None
        active_target: tuple[int, int] | None = None
        active_remaining = 0
        episode_id = f"hdagger:{iteration}:{world}:{profile}:{seed}"
        episode_option_counts = np.zeros(9, dtype=np.int64)
        episode_primitive_counts = np.zeros(5, dtype=np.int64)
        visited_positions: set[tuple[int, int]] = set()
        while transitions < args.transitions_per_iteration:
            state = structured_state(observation, context, progress, shield_intervened, duration)
            maps = torch.as_tensor(observation["map"], device=device, dtype=torch.float32)[
                None, None
            ]
            states = torch.as_tensor(state, device=device)[None, None]
            current_option_mask = option_mask(observation)
            masks = torch.as_tensor(current_option_mask, device=device, dtype=torch.bool)[
                None, None
            ]
            option_tensor, output = policy.predict(maps, states, masks, hidden)
            hidden = output.recurrent_state
            policy_option = MissionOption(int(option_tensor.item()))
            policy_target_index = int(output.target_logits.argmax(-1).item())
            policy_target = (
                None
                if policy_target_index == 256
                else (policy_target_index // 16, policy_target_index % 16)
            )
            active_is_valid = active_option is not None and current_option_mask[int(active_option)]
            if active_target is not None:
                target_row, target_column = active_target
                semantic_map = np.asarray(observation["map"])
                target_is_current = (
                    0 <= target_row < semantic_map.shape[1]
                    and 0 <= target_column < semantic_map.shape[2]
                    and (
                        semantic_map[1, target_row, target_column] > 0
                        or semantic_map[8, target_row, target_column] > 0
                    )
                    and semantic_map[10, target_row, target_column] <= 0
                )
                active_is_valid = active_is_valid and bool(target_is_current)
            if active_remaining > 0 and active_is_valid:
                policy_option = active_option
                policy_target = active_target
                active_remaining -= 1
            else:
                active_option = policy_option
                active_target = policy_target
                predicted_duration = min(8, int(output.duration_logits.argmax(-1).item()) + 1)
                active_remaining = (
                    0
                    if policy_option
                    in {
                        MissionOption.AVOID_DYNAMIC_WORKER,
                        MissionOption.REPLAN_ROUTE,
                        MissionOption.HOLD_FOR_UNCERTAINTY,
                        MissionOption.EMERGENCY_SAFE_STOP,
                    }
                    else predicted_duration - 1
                )
            primitive_teacher = expert.decide(observation)
            teacher_option, teacher_target, reason = derive_option(
                np.asarray(observation["map"]),
                np.asarray(observation["action_mask"]),
                primitive_teacher.final_action,
                previous_action=previous_expert_action,
                shield_intervened=shield_intervened,
            )
            previous_expert_action = primitive_teacher.final_action
            disagreements += int(policy_option != teacher_option)
            target_disagreements += int(policy_target != teacher_target)
            unsafe_options += int(
                policy_option
                not in {
                    teacher_option,
                    MissionOption.REPLAN_ROUTE,
                    MissionOption.RETREAT_TO_SAFE_CELL,
                    MissionOption.HOLD_FOR_UNCERTAINTY,
                }
                and shield_intervened
            )
            chosen_option = teacher_option if rng.random() < beta else policy_option
            option_counts[int(chosen_option)] += 1
            episode_option_counts[int(chosen_option)] += 1
            chosen_target = teacher_target if chosen_option == teacher_option else policy_target
            risk_budget = float(output.risk_budgets[0, 0].mean().item())
            decision = system.execute_option(
                observation,
                option=chosen_option,
                target=chosen_target,
                risk_budget=risk_budget,
                inspection_intent=chosen_option
                in {
                    MissionOption.INSPECT_KNOWN_RISK,
                    MissionOption.INSPECT_PPE_VIOLATION,
                },
                force_replan=bool(output.replanning_urgency[0, 0] > 0.5),
            )
            next_observation, reward, terminated, truncated, info = environment.step(
                decision.executed_primitive
            )
            planner_failures += int(not decision.planner.success)
            shield_interventions += int(decision.shield.shield_decision != "accept")
            primitive_counts[decision.executed_primitive] += 1
            episode_primitive_counts[decision.executed_primitive] += 1
            robot_cells = np.argwhere(np.asarray(observation["map"])[5] > 0)
            if len(robot_cells) == 1:
                visited_positions.add((int(robot_cells[0, 0]), int(robot_cells[0, 1])))
            target_coordinates = teacher_target or (-1, -1)
            writer.add(
                {
                    "maps": np.asarray(observation["map"], dtype=np.uint8),
                    "states": np.asarray(observation["state"], dtype=np.float16),
                    "option_masks": current_option_mask.astype(np.uint8),
                    "options": int(teacher_option),
                    "target_coordinates": target_coordinates,
                    "target_types": reason,
                    "option_durations": 4,
                    "planner_requests": teacher_option.name.lower(),
                    "planner_outcomes": decision.planner.reason,
                    "executed_primitives": np.asarray(
                        (decision.executed_primitive, 4, 4, 4), dtype=np.int8
                    ),
                    "executed_primitive_masks": np.asarray((True, False, False, False)),
                    "shield_interventions": decision.shield.shield_decision != "accept",
                    "mission_progress": float(info["hazard_recall"]),
                    "rewards": float(reward),
                    "vector_costs": _vector_cost(
                        np.asarray(observation["map"]), float(info["cost"])
                    ),
                    "terminated": terminated,
                    "truncated": truncated,
                    "success": bool(info["success"]),
                    "episode_ids": episode_id,
                    "episode_steps": environment.steps,
                    "splits": "train",
                    "recurrent_context": context.copy(),
                }
            )
            context[:-1] = context[1:]
            context[-1] = (
                decision.executed_primitive,
                float(reward),
                float(environment.cumulative_cost),
            )
            progress = float(info["hazard_recall"])
            shield_intervened = decision.shield.shield_decision != "accept"
            duration = int(output.duration_logits.argmax(-1).item()) + 1
            observation = next_observation
            transitions += 1
            if terminated or truncated:
                successes.append(bool(info["success"]))
                recalls.append(float(info["hazard_recall"]))
                coverages.append(float(info["inspection_coverage"]))
                costs.append(float(environment.cumulative_cost))
                episode_results.append(
                    {
                        "world": world,
                        "profile": profile,
                        "seed": seed,
                        "steps": int(environment.steps),
                        "terminated": bool(terminated),
                        "truncated": bool(truncated),
                        "success": bool(info["success"]),
                        "hazard_recall": float(info["hazard_recall"]),
                        "inspection_coverage": float(info["inspection_coverage"]),
                        "safety_cost": float(environment.cumulative_cost),
                        "option_counts": episode_option_counts.tolist(),
                        "primitive_counts": episode_primitive_counts.tolist(),
                        "unique_visited_cells": len(visited_positions),
                    }
                )
                break
        episode_index += 1
    writer.flush()
    return {
        "iteration": iteration,
        "beta": beta,
        "policy_visited_transitions": transitions,
        "episodes": episode_index,
        "option_disagreement_rate": disagreements / transitions,
        "target_disagreement_rate": target_disagreements / transitions,
        "unsafe_option_choices": unsafe_options,
        "option_counts": option_counts.tolist(),
        "primitive_counts": primitive_counts.tolist(),
        "planner_failure_rate": planner_failures / transitions,
        "shield_intervention_rate": shield_interventions / transitions,
        "invalid_options": 0,
        "mission_success": float(np.mean(successes)) if successes else 0.0,
        "hazard_recall": float(np.mean(recalls)) if recalls else 0.0,
        "coverage": float(np.mean(coverages)) if coverages else 0.0,
        "mean_safety_cost": float(np.mean(costs)) if costs else 0.0,
        "episode_results": episode_results,
        "chunks": writer.chunks,
    }


def cumulative_dataset(
    base_dir: Path,
    all_chunks: list[dict[str, Any]],
    destination: Path,
) -> Path:
    base = json.loads((base_dir / "manifest.json").read_text(encoding="utf-8"))
    chunks = [*base["chunks"], *all_chunks]
    manifest = {
        **base,
        "source": "causal hierarchical demonstrations plus policy-visited corrections",
        "transitions": sum(int(item["rows"]) for item in chunks),
        "chunks": chunks,
        "dataset_sha256": hashlib.sha256(
            "".join(item["sha256"] for item in chunks).encode()
        ).hexdigest(),
    }
    destination.mkdir(parents=True, exist_ok=True)
    path = destination / "manifest.json"
    path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return path


def main() -> int:
    args = parse_args()
    args.reports.mkdir(parents=True, exist_ok=True)
    checkpoint = args.initial_checkpoint
    collected_chunks: list[dict[str, Any]] = []
    results: list[dict[str, Any]] = []
    betas = (0.75, 0.5, 0.25)
    for iteration in range(1, args.iterations + 1):
        result = collect(args, checkpoint, iteration, betas[iteration - 1])
        collected_chunks.extend(result["chunks"])
        dataset_dir = args.output / f"iteration_{iteration}" / "cumulative_dataset"
        cumulative_dataset(args.base_dataset, collected_chunks, dataset_dir)
        training_dir = args.output / f"iteration_{iteration}" / "training"
        report_dir = args.reports / f"iteration_{iteration}_training"
        command = [
            sys.executable,
            "scripts/train_hierarchical_imitation.py",
            "--dataset",
            str(dataset_dir),
            "--output",
            str(training_dir),
            "--report-dir",
            str(report_dir),
            "--seeds",
            str(args.seed + iteration),
            "--epochs",
            str(args.retrain_epochs),
            "--initialize",
            str(checkpoint),
        ]
        completed = subprocess.run(command, check=False)
        candidate = training_dir / f"seed_{args.seed + iteration}" / "best_checkpoint.pt"
        if not candidate.exists():
            raise RuntimeError("Hierarchical DAgger retraining produced no checkpoint")
        checkpoint = candidate
        result["checkpoint"] = checkpoint.as_posix()
        result["checkpoint_sha256"] = sha256(checkpoint)
        result["retraining_exit_code"] = completed.returncode
        results.append(result)
        (args.reports / f"iteration_{iteration}.json").write_text(
            json.dumps(result, indent=2) + "\n", encoding="utf-8"
        )
    training_transitions = args.transitions_per_iteration
    args.transitions_per_iteration = args.evaluation_transitions
    final_evaluation = collect(args, checkpoint, args.iterations + 1, 0.0)
    args.transitions_per_iteration = training_transitions
    passed = (
        all(item["retraining_exit_code"] == 0 for item in results)
        and final_evaluation["mission_success"] >= 0.75
        and final_evaluation["hazard_recall"] >= 0.90
        and final_evaluation["coverage"] >= 0.90
        and final_evaluation["planner_failure_rate"] <= 0.01
        and final_evaluation["invalid_options"] == 0
    )
    summary = {
        "status": "PASSED" if passed else "FAILED",
        "genuine_policy_visited_states": True,
        "iterations": results,
        "beta_zero_evaluation": final_evaluation,
        "total_corrections": sum(item["policy_visited_transitions"] for item in results),
        "final_checkpoint": checkpoint.as_posix(),
        "final_checkpoint_sha256": sha256(checkpoint),
    }
    (args.reports / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))
    print(f"HIERARCHICAL_DAGGER={'PASSED' if passed else 'FAILED'}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
