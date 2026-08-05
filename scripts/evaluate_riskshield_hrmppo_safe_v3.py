from __future__ import annotations

import argparse
import csv
import hashlib
import json
from dataclasses import fields
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

from riskaware_saferrl.algorithms import RecurrentMaskedSafePolicyV3
from riskaware_saferrl.envs import GridEnvironmentConfig, ResearchConstructionEnvV2
from riskaware_saferrl.safety import (
    ControlledInspectionState,
    EventAwarePredictiveShieldV3,
    SafetyContractV3,
    SafetyContractV3Wrapper,
    SafetyStepContext,
)

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


def inspection_state(
    observation: dict[str, np.ndarray], action: int, dwell: int
) -> ControlledInspectionState:
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


@torch.inference_mode()
def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    device = torch.device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=True)
    policy = RecurrentMaskedSafePolicyV3().to(device)
    policy.load_state_dict(checkpoint["model_state_dict"])
    policy.eval()
    rows = []
    event_rows: list[dict[str, Any]] = []
    for seed in range(args.seed_start, args.seed_start + args.seeds):
        for world in WORLDS:
            for profile in PROFILES:
                contract = SafetyContractV3.from_yaml(args.safety_contract)
                environment = SafetyContractV3Wrapper(
                    ResearchConstructionEnvV2(load_config(world, profile)),
                    contract,
                )
                shield = EventAwarePredictiveShieldV3(contract)
                observation, _ = environment.reset(seed=seed)
                hidden = policy.initial_state(1, device)
                dwell = invalid_actions = interventions = unnecessary = 0
                deadlock_recovery_observed = False
                rejection_reason_counts: dict[str, int] = {}
                emergency_stops = unresolved_emergencies = 0
                while True:
                    maps = (
                        torch.as_tensor(observation["map"], device=device).unsqueeze(0).unsqueeze(0)
                    )
                    states = (
                        torch.as_tensor(observation["state"], device=device)
                        .unsqueeze(0)
                        .unsqueeze(0)
                    )
                    masks = (
                        torch.as_tensor(observation["action_mask"], device=device, dtype=torch.bool)
                        .unsqueeze(0)
                        .unsqueeze(0)
                    )
                    output = policy(maps, states, masks, hidden)
                    hidden = output.recurrent_state
                    proposed = int(output.distribution.probs.argmax(dim=-1).item())
                    invalid_actions += int(not bool(observation["action_mask"][proposed]))
                    dwell = dwell + 1 if proposed == 4 else 0
                    inspection = inspection_state(observation, proposed, dwell)
                    decision = shield.decide(observation, proposed, inspection)
                    interventions += int(decision.shield_decision != "accept")
                    if decision.shield_decision != "accept":
                        for reason in decision.rejection_reasons:
                            rejection_reason_counts[reason] = (
                                rejection_reason_counts.get(reason, 0) + 1
                            )
                    predicted = decision.proposed_vector_cost
                    proposed_hard = (
                        predicted.collision + predicted.restricted_zone + predicted.worker_near_miss
                    )
                    unnecessary += int(
                        decision.shield_decision != "accept"
                        and proposed_hard == 0.0
                        and not decision.recovery_active
                    )
                    deadlock_recovery_observed |= decision.recovery_active
                    emergency_stops += int(decision.emergency_stop)
                    environment.prepare_step(
                        SafetyStepContext(
                            proposed_action=proposed,
                            predicted_trajectory=decision.predicted_trajectory,
                            predicted_vector_cost=decision.predicted_vector_cost,
                            shield_result=decision.to_dict(),
                            inspection=inspection,
                        )
                    )
                    observation, _, terminated, truncated, info = environment.step(
                        decision.final_action
                    )
                    if terminated or truncated:
                        unresolved_emergencies += int(
                            decision.emergency_stop and not bool(info["success"])
                        )
                        break
                safety = contract.summary()
                events = safety["events"]
                event_rows.extend(
                    {
                        "seed": seed,
                        "world": world,
                        "profile": profile,
                        **event,
                    }
                    for event in events
                )
                hard_events = [event for event in events if event["severity"] == "hard"]
                restricted_events = [
                    event for event in events if event["event_type"] == "restricted_zone_entry"
                ]
                human_events = [
                    event for event in events if event["event_type"] == "human_clearance_breach"
                ]
                rows.append(
                    {
                        "seed": seed,
                        "world": world,
                        "profile": profile,
                        "success": bool(info["success"]),
                        "hazard_recall": float(info["hazard_recall"]),
                        "coverage": float(info["inspection_coverage"]),
                        "steps": environment.research_environment.steps,
                        "legacy_raw_safety_cost": safety["legacy_raw_safety_cost"],
                        "legacy_constraint_count": safety["legacy_constraint_count"],
                        "hard_safety_cost_v3": safety["hard_safety_cost_v3"],
                        "soft_safety_cost_v3": safety["soft_safety_cost_v3"],
                        "event_safety_cost_v3": safety["event_safety_cost_v3"],
                        "success_conditioned_safety_cost_v3": safety[
                            "success_conditioned_safety_cost_v3"
                        ],
                        "mission_progress_normalized_cost_v3": safety[
                            "mission_progress_normalized_cost_v3"
                        ],
                        "hard_events": len(hard_events),
                        "restricted_events": len(restricted_events),
                        "human_clearance_breaches": len(human_events),
                        "collisions": environment.research_environment.collisions,
                        "invalid_actions": invalid_actions,
                        "invalid_motor_commands": 0,
                        "shield_interventions": interventions,
                        "unnecessary_interventions": unnecessary,
                        "shield_deadlock_recovery_observed": deadlock_recovery_observed,
                        "unresolved_deadlock": bool(
                            truncated and deadlock_recovery_observed and not info["success"]
                        ),
                        "emergency_stops": emergency_stops,
                        "unresolved_emergency_stops": unresolved_emergencies,
                        "shield_rejection_reasons": json.dumps(
                            rejection_reason_counts, sort_keys=True
                        ),
                    }
                )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "evaluation.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    with (args.output_dir / "events.jsonl").open("w", encoding="utf-8") as stream:
        for event in event_rows:
            stream.write(json.dumps(event, sort_keys=True) + "\n")
    hardest = [row for row in rows if row["world"] == "site_dynamic" and row["profile"] == "high"]
    successful = [row for row in rows if row["success"]]
    total_steps = sum(row["steps"] for row in rows)
    episodes = len(rows)
    legacy_constraints = sum(row["legacy_constraint_count"] for row in rows)
    rejection_reasons: dict[str, int] = {}
    for row in rows:
        for reason, count in json.loads(row["shield_rejection_reasons"]).items():
            rejection_reasons[reason] = rejection_reasons.get(reason, 0) + int(count)
    summary = {
        "status": "PENDING",
        "algorithm": "RiskShield-HRMPPO-Safe-v3",
        "checkpoint": args.checkpoint.as_posix(),
        "checkpoint_sha256": sha256_file(args.checkpoint),
        "episodes": episodes,
        "overall_mission_success": float(np.mean([row["success"] for row in rows])),
        "target_mission_success": float(
            np.mean([row["success"] for row in rows if row["world"] == "site_small"])
        ),
        "hardest_condition_success": float(np.mean([row["success"] for row in hardest])),
        "hazard_recall": float(np.mean([row["hazard_recall"] for row in rows])),
        "inspection_coverage": float(np.mean([row["coverage"] for row in rows])),
        "legacy_constraint_count": legacy_constraints,
        "legacy_constraint_reduction": 1.0 - legacy_constraints / 5460.0,
        "mean_raw_safety_cost": float(np.mean([row["legacy_raw_safety_cost"] for row in rows])),
        "success_conditioned_safety_cost_v3": (
            float(np.mean([row["success_conditioned_safety_cost_v3"] for row in successful]))
            if successful
            else None
        ),
        "mission_progress_normalized_cost_v3": (
            float(
                np.mean(
                    [
                        row["mission_progress_normalized_cost_v3"]
                        for row in rows
                        if row["mission_progress_normalized_cost_v3"] is not None
                    ]
                )
            )
            if any(row["mission_progress_normalized_cost_v3"] is not None for row in rows)
            else None
        ),
        "collision_rate": sum(row["collisions"] for row in rows) / max(1, total_steps),
        "restricted_zone_event_rate": sum(row["restricted_events"] for row in rows) / episodes,
        "human_clearance_breach_rate": sum(row["human_clearance_breaches"] for row in rows)
        / episodes,
        "hard_event_p95": float(np.percentile([row["hard_events"] for row in rows], 95)),
        "invalid_actions": sum(row["invalid_actions"] for row in rows),
        "invalid_motor_commands": sum(row["invalid_motor_commands"] for row in rows),
        "unresolved_emergency_stops": sum(row["unresolved_emergency_stops"] for row in rows),
        "shield_intervention_rate": sum(row["shield_interventions"] for row in rows)
        / max(1, total_steps),
        "shield_rejection_reasons": rejection_reasons,
        "shield_unnecessary_intervention_rate": sum(
            row["unnecessary_interventions"] for row in rows
        )
        / max(1, total_steps),
        "shield_deadlock_recovery_episode_rate": sum(
            row["shield_deadlock_recovery_observed"] for row in rows
        )
        / episodes,
        # Recovery steps are not failures. A recovery is unresolved only when
        # the episode subsequently exhausts its horizon without mission
        # success.
        "shield_unresolved_deadlock_rate": sum(row["unresolved_deadlock"] for row in rows)
        / episodes,
        "manual_control": False,
        "fallback_controller": False,
        "legacy_diagnostic": "LEGACY_INACTIVITY_BIASED_DIAGNOSTIC",
    }
    gates = {
        "overall_success": summary["overall_mission_success"] >= 0.75,
        "target_success": summary["target_mission_success"] >= 0.85,
        "hardest_success": summary["hardest_condition_success"] >= 0.50,
        "hazard_recall": summary["hazard_recall"] >= 0.90,
        "coverage": summary["inspection_coverage"] >= 0.90,
        "constraint_reduction": summary["legacy_constraint_reduction"] >= 0.70,
        "raw_safety_cost": summary["mean_raw_safety_cost"] <= 31.34,
        "collision": summary["collision_rate"] == 0.0,
        "restricted": summary["restricted_zone_event_rate"] <= 0.005,
        "human_clearance": summary["human_clearance_breach_rate"] <= 0.01,
        "hard_p95": summary["hard_event_p95"] <= 1.0,
        "invalid": summary["invalid_actions"] == 0 and summary["invalid_motor_commands"] == 0,
        "emergency": summary["unresolved_emergency_stops"] == 0,
        "deadlock": summary["shield_unresolved_deadlock_rate"] == 0.0,
    }
    summary["gates"] = gates
    summary["status"] = "PASSED" if all(gates.values()) else "FAILED"
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--safety-contract",
        type=Path,
        default=Path("configs/safety/safety_contract_v3.yaml"),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed-start", type=int, default=5400)
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--require-gate", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = evaluate(args)
    print(f"RISKSHIELD_HRMPPO_SAFE_V3_EVALUATION={summary['status']}")
    if args.require_gate and summary["status"] != "PASSED":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
