from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from run_dagger_v2 import (
    PROFILES,
    WORLDS,
    cumulative_manifest,
    evaluate_policy,
    load_config,
    load_policy,
    policy_action,
    sha256_file,
    train_iteration,
)

from riskaware_saferrl.baselines import CausalObservationExpert
from riskaware_saferrl.safety import (
    ControlledInspectionState,
    EventAwarePredictiveShieldV3,
    SafetyContractV3,
)
from riskaware_saferrl.training.safety_dagger import (
    SafetyCorrectionWriter,
    select_safety_state,
)


def inspection_state(observation: dict[str, np.ndarray], action: int, dwell: int):
    robot = np.argwhere(observation["map"][5] > 0)[0]
    hazards = np.argwhere(observation["map"][1] > 0)
    workers = np.argwhere(observation["map"][2] > 0)
    recognized = any(
        abs(int(robot[0]) - int(target[0])) + abs(int(robot[1]) - int(target[1])) <= 2
        for target in hazards
    )
    clearance = (
        min(
            abs(int(robot[0]) - int(worker[0])) + abs(int(robot[1]) - int(worker[1]))
            for worker in workers
        )
        if len(workers)
        else float("inf")
    )
    return ControlledInspectionState(
        target_recognized=recognized,
        inspection_intent=action == 4,
        safe_approach=clearance >= 2,
        speed=0.0 if action == 4 else 1.0,
        human_clearance_cells=float(clearance),
        dwell_steps=dwell,
        retreat_route_available=bool(observation["action_mask"][:4].any()),
        shield_active=True,
    )


def collect_iteration(
    *,
    checkpoint: Path,
    device: torch.device,
    iteration: int,
    beta: float,
    target_corrections: int,
    seed: int,
    output_dir: Path,
    chunk_size: int,
    task_anchor_probability: float,
    expert_strategy: str,
    contiguous_trajectories: bool,
) -> dict[str, Any]:
    rng = random.Random(seed)
    policy = load_policy(checkpoint, device)
    writer = SafetyCorrectionWriter(output_dir / "chunks", chunk_size)
    selected = visited = disagreements = invalid = 0
    episodes: list[dict[str, Any]] = []
    reason_counts: dict[str, int] = {}
    episode_index = 0
    while selected < target_corrections:
        world = WORLDS[episode_index % len(WORLDS)]
        profile = tuple(PROFILES)[(episode_index // len(WORLDS)) % len(PROFILES)]
        from riskaware_saferrl.envs import ResearchConstructionEnvV2

        environment = ResearchConstructionEnvV2(load_config(world, profile))
        environment_seed = seed + episode_index
        observation, _ = environment.reset(seed=environment_seed)
        expert = CausalObservationExpert(planning_strategy=expert_strategy)
        contract = SafetyContractV3.from_yaml(Path("configs/safety/safety_contract_v3.yaml"))
        policy_shield = EventAwarePredictiveShieldV3(contract)
        teacher_shield = EventAwarePredictiveShieldV3(contract)
        hidden = None
        dwell = 0
        recent_positions: list[tuple[int, int]] = []
        episode_selected = 0
        episode_id = f"safety-dagger:{iteration}:{world}:{profile}:{environment_seed}"
        while selected < target_corrections:
            proposed, hidden = policy_action(policy, observation, hidden, device)
            expert_proposal = int(expert.decide(observation).final_action)
            dwell = dwell + 1 if expert_proposal == 4 else 0
            policy_inspection = inspection_state(observation, proposed, dwell)
            teacher_inspection = inspection_state(observation, expert_proposal, dwell)
            policy_decision = policy_shield.decide(observation, proposed, policy_inspection)
            teacher_decision = teacher_shield.decide(
                observation, expert_proposal, teacher_inspection
            )
            teacher_action = int(teacher_decision.final_action)
            if not bool(observation["action_mask"][proposed]):
                invalid += 1
                raise RuntimeError("Masked policy emitted an invalid action")
            if not bool(observation["action_mask"][teacher_action]):
                invalid += 1
                raise RuntimeError("Safety teacher emitted an invalid action")
            position = tuple(int(value) for value in np.argwhere(observation["map"][5] > 0)[0])
            recent_positions.append(position)
            recent_positions = recent_positions[-12:]
            selection = select_safety_state(
                policy_action=proposed,
                teacher_action=teacher_action,
                policy_shield=policy_decision,
                recent_positions=tuple(recent_positions),
                include_safe_example=rng.random() < task_anchor_probability,
            )
            disagreements += int(proposed != teacher_action)
            visited += 1
            should_store = selection.selected or contiguous_trajectories
            if should_store:
                stored_reasons = selection.reasons if selection.reasons else ("mission_context",)
                writer.add(
                    {
                        "map": observation["map"],
                        "state": observation["state"],
                        "action_mask": observation["action_mask"],
                        "teacher_action": teacher_action,
                        "policy_action": proposed,
                        "shield_action": policy_decision.final_action,
                        "episode_id": episode_id,
                        "selection_reasons": stored_reasons,
                        "selection_priority": selection.priority,
                        "predicted_vector_cost": (policy_decision.proposed_vector_cost.as_dict()),
                    }
                )
                selected += 1
                episode_selected += 1
                for reason in stored_reasons:
                    reason_counts[reason] = reason_counts.get(reason, 0) + 1
            executed = teacher_action if rng.random() < beta else proposed
            observation, _, terminated, truncated, info = environment.step(executed)
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
                "selected": episode_selected,
            }
        )
        episode_index += 1
    writer.flush()
    return {
        "iteration": iteration,
        "beta": beta,
        "policy_visited_transitions": visited,
        "new_visited_state_transitions": selected,
        "episodes": len(episodes),
        "disagreement_count": disagreements,
        "disagreement_rate": disagreements / max(1, visited),
        "invalid_actions": invalid,
        "selection_reason_counts": reason_counts,
        "action_counts": writer.action_counts.tolist(),
        "chunks": writer.chunks,
        "collection_success_rate": float(np.mean([row["success"] for row in episodes])),
        "collection_hazard_recall": float(np.mean([row["hazard_recall"] for row in episodes])),
        "collection_coverage": float(np.mean([row["coverage"] for row in episodes])),
        "collection_safety_cost": float(np.mean([row["safety_cost"] for row in episodes])),
        "collection_collision_rate": sum(row["collisions"] for row in episodes)
        / max(1, sum(row["steps"] for row in episodes)),
        "observation_only_teacher": True,
        "expert_strategy": expert_strategy,
        "contiguous_trajectories": contiguous_trajectories,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base-dataset",
        type=Path,
        default=Path("artifacts/strong_policy_upgrade/causal_expert_demonstrations_systematic"),
    )
    parser.add_argument(
        "--initial-checkpoint",
        type=Path,
        default=Path(
            "artifacts/strong_policy_upgrade/dagger/iteration_2/training/best_behavior_cloning.pt"
        ),
    )
    parser.add_argument(
        "--artifact-dir",
        type=Path,
        default=Path("artifacts/strong_policy_upgrade/safety_dagger"),
    )
    parser.add_argument(
        "--report-dir",
        type=Path,
        default=Path("reports/strong_policy_upgrade/safety_dagger"),
    )
    parser.add_argument("--corrections-per-iteration", type=int, default=17_000)
    parser.add_argument("--chunk-size", type=int, default=4_000)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--held-out-seeds", type=int, default=10)
    parser.add_argument("--task-anchor-probability", type=float, default=0.20)
    parser.add_argument("--anchor-kl-coefficient", type=float, default=2.0)
    parser.add_argument(
        "--expert-strategy",
        choices=("systematic", "risk_astar"),
        default="systematic",
    )
    parser.add_argument(
        "--contiguous-trajectories",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Store complete policy-visited context instead of fake-adjacent selected states.",
    )
    parser.add_argument("--seed", type=int, default=20260730)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    args.report_dir.mkdir(parents=True, exist_ok=True)
    current_checkpoint = args.initial_checkpoint
    correction_chunks: list[dict[str, Any]] = []
    results: list[dict[str, Any]] = []
    for iteration, beta in enumerate((0.75, 0.50, 0.25), start=1):
        iteration_artifacts = args.artifact_dir / f"iteration_{iteration}"
        report_path = args.report_dir / f"iteration_{iteration}.json"
        if iteration_artifacts.exists() or report_path.exists():
            raise FileExistsError(f"Refusing to overwrite safety DAgger iteration {iteration}")
        collection = collect_iteration(
            checkpoint=current_checkpoint,
            device=device,
            iteration=iteration,
            beta=beta,
            target_corrections=args.corrections_per_iteration,
            seed=args.seed + 10_000 * iteration,
            output_dir=iteration_artifacts,
            chunk_size=args.chunk_size,
            task_anchor_probability=args.task_anchor_probability,
            expert_strategy=args.expert_strategy,
            contiguous_trajectories=args.contiguous_trajectories,
        )
        dataset_dir = iteration_artifacts / "cumulative_dataset"
        manifest_path = cumulative_manifest(
            args.base_dataset / "manifest.json",
            correction_chunks,
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
            anchor_kl_coefficient=args.anchor_kl_coefficient,
        )
        training_summary_path = iteration_artifacts / "training" / "behavior_cloning_summary.json"
        training_summary = json.loads(training_summary_path.read_text(encoding="utf-8"))
        if training_summary["status"] != "PASSED":
            failure = {
                **collection,
                "status": "FAILED",
                "failure_stage": "behavior_cloning_gate",
                "behavior_cloning": training_summary,
                "rejected_checkpoint_path": current_checkpoint.as_posix(),
                "rejected_checkpoint_sha256": sha256_file(current_checkpoint),
            }
            report_path.write_text(json.dumps(failure, indent=2) + "\n", encoding="utf-8")
            raise RuntimeError(f"Safety DAgger iteration {iteration} failed Behavior Cloning gates")
        evaluation = evaluate_policy(
            load_policy(current_checkpoint, device),
            device,
            range(5000, 5000 + args.held_out_seeds),
        )
        result = {
            **collection,
            "cumulative_dataset_manifest": manifest_path.as_posix(),
            "checkpoint_path": current_checkpoint.as_posix(),
            "checkpoint_sha256": sha256_file(current_checkpoint),
            "held_out": evaluation,
        }
        report_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        results.append(result)
        correction_chunks.extend(collection["chunks"])
    total = sum(row["new_visited_state_transitions"] for row in results)
    final_evaluation = evaluate_policy(
        load_policy(current_checkpoint, device),
        device,
        range(6000, 6000 + args.held_out_seeds),
    )
    integrity_gate = (
        total >= 50_000
        and all(row["invalid_actions"] == 0 for row in results)
        and final_evaluation["invalid_actions"] == 0
    )
    task_competence_gate = (
        final_evaluation["success_rate"] >= 0.75
        and final_evaluation["hazard_recall"] >= 0.90
        and final_evaluation["coverage"] >= 0.90
    )
    status = "PASSED" if integrity_gate and task_competence_gate else "FAILED"
    summary = {
        "status": status,
        "genuine_policy_visited_states": True,
        "observation_only_safety_teacher": True,
        "iterations": results,
        "total_safety_corrections": total,
        "integrity_gate": integrity_gate,
        "task_competence_gate": task_competence_gate,
        "final_checkpoint": current_checkpoint.as_posix(),
        "final_checkpoint_sha256": sha256_file(current_checkpoint),
        "final_evaluation": final_evaluation,
    }
    (args.report_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(f"SAFETY_DAGGER={status}")
    if status != "PASSED":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
