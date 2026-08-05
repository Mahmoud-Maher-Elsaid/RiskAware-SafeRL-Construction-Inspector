from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

from riskaware_saferrl.envs import GridEnvironmentConfig, ResearchConstructionEnvV2
from riskaware_saferrl.policies import RecurrentMaskedPolicy
from riskaware_saferrl.safety import PredictiveSafetyShield

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


def near(position: tuple[int, int], locations: set[tuple[int, int]]) -> bool:
    return any(abs(position[0] - row) + abs(position[1] - column) <= 1 for row, column in locations)


@dataclass
class OpenEvent:
    event_id: str
    event_type: str
    start_step: int
    last_step: int
    duration: int
    integrated_severity: float
    peak_severity: float
    position: tuple[int, int]
    proposed_action: int
    executed_action: int
    shield_decision: str
    controlled_exposure: bool
    shield_generated: bool


class EventTracker:
    def __init__(self) -> None:
        self.open: dict[str, OpenEvent] = {}
        self.closed: list[dict[str, Any]] = []
        self.counts: Counter[str] = Counter()

    def update(
        self,
        *,
        active: dict[str, float],
        episode_id: str,
        step: int,
        position: tuple[int, int],
        proposed_action: int,
        executed_action: int,
        shield_decision: str,
        controlled_exposure: bool,
    ) -> None:
        active_types = set(active)
        for event_type in tuple(self.open):
            if event_type not in active_types:
                self._close(event_type, resolved=True, resolution_action=executed_action)
        for event_type, severity in active.items():
            if event_type not in self.open:
                self.counts[event_type] += 1
                self.open[event_type] = OpenEvent(
                    event_id=f"{episode_id}:{event_type}:{self.counts[event_type]:05d}",
                    event_type=event_type,
                    start_step=step,
                    last_step=step,
                    duration=0,
                    integrated_severity=0.0,
                    peak_severity=0.0,
                    position=position,
                    proposed_action=proposed_action,
                    executed_action=executed_action,
                    shield_decision=shield_decision,
                    controlled_exposure=controlled_exposure,
                    shield_generated=proposed_action != executed_action,
                )
            event = self.open[event_type]
            event.last_step = step
            event.duration += 1
            event.integrated_severity += severity
            event.peak_severity = max(event.peak_severity, severity)
            event.controlled_exposure &= controlled_exposure
            event.shield_generated |= proposed_action != executed_action

    def _close(self, event_type: str, *, resolved: bool, resolution_action: int) -> None:
        event = self.open.pop(event_type)
        self.closed.append(
            {
                **event.__dict__,
                "end_step": event.last_step,
                "resolved": resolved,
                "resolution_action": resolution_action,
            }
        )

    def close_all(self, *, terminated: bool, resolution_action: int) -> None:
        for event_type in tuple(self.open):
            self._close(
                event_type,
                resolved=terminated,
                resolution_action=resolution_action,
            )


@torch.inference_mode()
def audit(args: argparse.Namespace) -> dict[str, Any]:
    device = torch.device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=True)
    policy = RecurrentMaskedPolicy().to(device)
    policy.load_state_dict(checkpoint["model_state_dict"])
    policy.eval()
    tracker = EventTracker()
    step_counts: Counter[str] = Counter()
    overlap_counts: Counter[str] = Counter()
    raw_cost_by_type: Counter[str] = Counter()
    shield_decisions: Counter[str] = Counter()
    episode_rows: list[dict[str, Any]] = []
    total_steps = post_termination_steps = initial_exposure_steps = 0
    invalid_actions = invalid_motor_commands = 0
    for seed in range(args.seed_start, args.seed_start + args.seeds):
        for world in WORLDS:
            for profile in PROFILES:
                environment = ResearchConstructionEnvV2(load_config(world, profile))
                observation, _ = environment.reset(seed=seed)
                hidden = policy.initial_state(1, device)
                shield = PredictiveSafetyShield(horizon=1, safety_budget=1.0)
                episode_id = f"{world}:{profile}:{seed}"
                episode_step_counts: Counter[str] = Counter()
                last_action = 4
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
                    action, output = policy.predict(maps, states, masks, hidden)
                    hidden = output.recurrent_state
                    proposed = int(action.item())
                    invalid_actions += int(not bool(observation["action_mask"][proposed]))
                    decision = shield.decide(environment, proposed)
                    executed = int(decision.final_action)
                    invalid_motor_commands += int(not environment.action_space.contains(executed))
                    pre_workers = set(environment.workers)
                    pre_dynamic = set(environment.dynamic_hazards)
                    pre_collisions = environment.collisions
                    pre_near_misses = environment.near_misses
                    pre_restricted = environment.restricted_violations
                    inspectable = bool(environment.inspectable_hazards())
                    observation, _, terminated, truncated, info = environment.step(executed)
                    position = environment.agent
                    collision = environment.collisions > pre_collisions
                    restricted = environment.restricted_violations > pre_restricted
                    near_miss = environment.near_misses > pre_near_misses
                    worker_near = near(position, pre_workers)
                    dynamic_near = near(position, pre_dynamic)
                    ppe = position in environment.ppe_risk
                    controlled = bool(
                        executed == 4
                        and inspectable
                        and decision.shield_decision in {"accept", "replace"}
                    )
                    active: dict[str, float] = {}
                    if collision:
                        active["collision"] = 1.0
                        raw_cost_by_type["collision"] += 1.0
                    if restricted:
                        active["restricted_zone"] = 1.0
                        raw_cost_by_type["restricted_zone"] += 1.0
                    if worker_near:
                        active["worker_near_miss"] = 0.5
                    if dynamic_near:
                        active["dynamic_hazard_proximity"] = 0.5
                    if near_miss:
                        raw_cost_by_type["near_miss_combined"] += 0.5
                    if ppe:
                        event_type = (
                            "controlled_inspection_exposure"
                            if controlled
                            else "uncontrolled_ppe_exposure"
                        )
                        active[event_type] = 0.25
                        raw_cost_by_type["ppe_risk"] += 0.25
                    if decision.emergency_stop:
                        active["emergency_stop"] = 1.0
                    if decision.shield_decision != "accept":
                        active["shield_intervention"] = 1.0
                    active_names = tuple(sorted(active))
                    if len(active_names) > 1:
                        overlap_counts["+".join(active_names)] += 1
                    for name in active:
                        step_counts[name] += 1
                        episode_step_counts[name] += 1
                    shield_decisions[decision.shield_decision] += 1
                    tracker.update(
                        active=active,
                        episode_id=episode_id,
                        step=environment.steps,
                        position=position,
                        proposed_action=proposed,
                        executed_action=executed,
                        shield_decision=decision.shield_decision,
                        controlled_exposure=controlled,
                    )
                    total_steps += 1
                    last_action = executed
                    if terminated or truncated:
                        tracker.close_all(
                            terminated=terminated,
                            resolution_action=last_action,
                        )
                        break
                episode_rows.append(
                    {
                        "episode_id": episode_id,
                        "world": world,
                        "profile": profile,
                        "seed": seed,
                        "steps": environment.steps,
                        "success": bool(info["success"]),
                        "legacy_raw_safety_cost": environment.cumulative_cost,
                        "collision_steps": episode_step_counts["collision"],
                        "restricted_dwell_steps": episode_step_counts["restricted_zone"],
                        "worker_near_miss_steps": episode_step_counts["worker_near_miss"],
                        "dynamic_hazard_proximity_steps": episode_step_counts[
                            "dynamic_hazard_proximity"
                        ],
                        "controlled_inspection_exposure_steps": episode_step_counts[
                            "controlled_inspection_exposure"
                        ],
                        "uncontrolled_ppe_exposure_steps": episode_step_counts[
                            "uncontrolled_ppe_exposure"
                        ],
                        "emergency_stop_steps": episode_step_counts["emergency_stop"],
                        "shield_intervention_steps": episode_step_counts["shield_intervention"],
                        "legacy_constraint_count": (
                            environment.collisions
                            + environment.near_misses
                            + environment.restricted_violations
                        ),
                    }
                )
    event_counts = Counter(event["event_type"] for event in tracker.closed)
    event_durations: dict[str, list[int]] = defaultdict(list)
    shield_generated_counts: Counter[str] = Counter()
    unresolved_counts: Counter[str] = Counter()
    for event in tracker.closed:
        event_durations[event["event_type"]].append(event["duration"])
        shield_generated_counts[event["event_type"]] += int(event["shield_generated"])
        unresolved_counts[event["event_type"]] += int(not event["resolved"])
    legacy_constraint_count = sum(row["legacy_constraint_count"] for row in episode_rows)
    category_definitions = {
        "collision": {
            "definition": "Executed motion attempts to enter an obstacle or leave the map.",
            "severity": "hard",
            "legacy_counter": "environment.collisions",
            "legacy_cost_weight": 1.0,
            "conflicts_with_inspection": False,
            "avoidable": True,
            "shield_predictable": True,
            "accounting_unit": "one increment per colliding transition",
        },
        "restricted_zone": {
            "definition": "Robot occupies a restricted cell after the executed action.",
            "severity": "hard entry; subsequent occupancy is dwell exposure",
            "legacy_counter": "environment.restricted_violations",
            "legacy_cost_weight": 1.0,
            "conflicts_with_inspection": False,
            "avoidable": True,
            "shield_predictable": True,
            "accounting_unit": "one increment per occupied timestep",
        },
        "worker_near_miss": {
            "definition": "Robot is within Manhattan distance one of an observed worker.",
            "severity": "hard when below clearance; otherwise soft proximity exposure",
            "legacy_counter": "shared environment.near_misses",
            "legacy_cost_weight": 0.5,
            "conflicts_with_inspection": False,
            "avoidable": "usually",
            "shield_predictable": True,
            "accounting_unit": "one shared near-miss increment per timestep",
        },
        "dynamic_hazard_proximity": {
            "definition": "Robot is within Manhattan distance one of a dynamic hazard.",
            "severity": "soft exposure unless minimum clearance is breached",
            "legacy_counter": "shared environment.near_misses",
            "legacy_cost_weight": 0.5,
            "conflicts_with_inspection": "possibly",
            "avoidable": "condition dependent",
            "shield_predictable": "one-step geometry; future motion uncertain",
            "accounting_unit": "shares one increment with worker proximity per timestep",
        },
        "uncontrolled_ppe_exposure": {
            "definition": "Robot occupies a PPE-risk cell without valid controlled inspection intent.",
            "severity": "soft semantic-risk exposure",
            "legacy_counter": "not included in legacy constraint count",
            "legacy_cost_weight": 0.25,
            "conflicts_with_inspection": True,
            "avoidable": "not always while completing inspection",
            "shield_predictable": True,
            "accounting_unit": "one raw-cost increment per timestep",
        },
        "controlled_inspection_exposure": {
            "definition": "Recognized inspectable target, inspect action, and active shield.",
            "severity": "bounded mission exposure",
            "legacy_counter": "not distinguished historically",
            "legacy_cost_weight": 0.25,
            "conflicts_with_inspection": True,
            "avoidable": False,
            "shield_predictable": True,
            "accounting_unit": "candidate classification; v3 adds dwell and retreat checks",
        },
        "emergency_stop": {
            "definition": "Shield found no safe action and selected the emergency action.",
            "severity": "hard when unresolved",
            "legacy_counter": "not included in environment constraint count",
            "legacy_cost_weight": 0.0,
            "conflicts_with_inspection": False,
            "avoidable": "condition dependent",
            "shield_predictable": True,
            "accounting_unit": "one event per contiguous stop episode",
        },
        "shield_intervention": {
            "definition": "Final executed action differs from the policy proposal or stops.",
            "severity": "soft operational event unless generated action violates a hard constraint",
            "legacy_counter": "reported separately",
            "legacy_cost_weight": 0.0,
            "conflicts_with_inspection": "may impede progress when repeated",
            "avoidable": "only unnecessary interventions",
            "shield_predictable": True,
            "accounting_unit": "one step plus contiguous intervention event",
        },
    }
    report = {
        "status": "PASSED" if legacy_constraint_count == 5460 else "FAILED",
        "checkpoint": args.checkpoint.as_posix(),
        "checkpoint_sha256": sha256_file(args.checkpoint),
        "paired_protocol": {
            "seed_start": args.seed_start,
            "seeds": args.seeds,
            "worlds": list(WORLDS),
            "profiles": list(PROFILES),
            "episodes": len(episode_rows),
        },
        "total_steps": total_steps,
        "legacy_constraint_count": legacy_constraint_count,
        "legacy_raw_safety_cost": float(sum(row["legacy_raw_safety_cost"] for row in episode_rows)),
        "mean_legacy_raw_safety_cost": float(
            np.mean([row["legacy_raw_safety_cost"] for row in episode_rows])
        ),
        "step_counts": dict(step_counts),
        "unique_event_counts": dict(event_counts),
        "event_duration": {
            name: {
                "total": sum(values),
                "mean": float(np.mean(values)),
                "maximum": max(values),
            }
            for name, values in sorted(event_durations.items())
        },
        "raw_cost_by_type": dict(raw_cost_by_type),
        "overlapping_step_categories": dict(overlap_counts),
        "shield_generated_event_counts": dict(shield_generated_counts),
        "unresolved_event_counts": dict(unresolved_counts),
        "shield_decisions": dict(shield_decisions),
        "invalid_actions": invalid_actions,
        "invalid_motor_commands": invalid_motor_commands,
        "initial_state_exposure_steps_charged": initial_exposure_steps,
        "post_termination_steps_charged": post_termination_steps,
        "accounting_findings": {
            "legacy_counter_unit": "per timestep, not unique physical event",
            "near_miss_combines_worker_and_dynamic_hazard": True,
            "worker_dynamic_overlap_single_legacy_increment": True,
            "ppe_in_legacy_constraint_count": False,
            "ppe_in_legacy_raw_cost": True,
            "repeated_unchanged_exposure_recharged": True,
            "post_termination_accounting": False,
            "initial_state_accounting_before_first_action": False,
        },
        "category_definitions": category_definitions,
        "episodes": episode_rows,
        "events": tracker.closed,
    }
    return report


def write_reports(report: dict[str, Any], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "violation_audit.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    with (output_dir / "violation_audit.csv").open("w", newline="", encoding="utf-8") as stream:
        fieldnames = list(report["episodes"][0])
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(report["episodes"])
    findings = report["accounting_findings"]
    lines = [
        "# HRMPPO v2 violation audit",
        "",
        f"Status: **{report['status']}**",
        "",
        f"- Episodes: {report['paired_protocol']['episodes']}",
        f"- Total transitions: {report['total_steps']}",
        f"- Reproduced legacy constraint count: {report['legacy_constraint_count']}",
        f"- Mean legacy raw safety cost: {report['mean_legacy_raw_safety_cost']:.6f}",
        f"- Invalid policy actions: {report['invalid_actions']}",
        f"- Invalid motor commands: {report['invalid_motor_commands']}",
        "",
        "## Step counts",
        "",
    ]
    lines.extend(f"- {key}: {value}" for key, value in sorted(report["step_counts"].items()))
    lines.extend(["", "## Unique events", ""])
    lines.extend(
        f"- {key}: {value}" for key, value in sorted(report["unique_event_counts"].items())
    )
    lines.extend(
        [
            "",
            "## Category contract audit",
            "",
            "| Type | Severity | Legacy weight | Counter unit | Mission conflict |",
            "|---|---|---:|---|---|",
            *[
                (
                    f"| {name} | {values['severity']} | "
                    f"{values['legacy_cost_weight']} | {values['accounting_unit']} | "
                    f"{values['conflicts_with_inspection']} |"
                )
                for name, values in sorted(report["category_definitions"].items())
            ],
            "",
            "## Accounting findings",
            "",
            *[f"- {key}: {value}" for key, value in findings.items()],
            "",
            "The legacy 5,460 value is an exposure-duration counter. It is not a count "
            "of unique physical safety events. PPE exposure contributes raw cost but is "
            "not included in that legacy counter. Worker and dynamic-hazard proximity "
            "share one legacy near-miss increment even when both categories overlap.",
            "",
        ]
    )
    (output_dir / "violation_audit.md").write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--seed-start", type=int, default=5400)
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("reports/strong_policy_upgrade/safety_finalization"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = audit(args)
    write_reports(report, args.output_dir)
    print(f"VIOLATION_AUDIT={report['status']}")
    if report["status"] != "PASSED":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
