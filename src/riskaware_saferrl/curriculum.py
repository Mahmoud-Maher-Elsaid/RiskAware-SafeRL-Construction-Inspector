from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from riskaware_saferrl.evaluation.expert_baselines import evaluate_plan
from riskaware_saferrl.planners import build_viewpoint_inspection_plan
from riskaware_saferrl.scenarios import Scenario

CURRICULUM_SCHEMA_VERSION = 1
CURRICULUM_TIERS = ("easy", "medium", "hard")


def sha256_file(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def evaluate_curriculum_scenario(
    scenario: Scenario,
    *,
    inspection_radius: int,
) -> dict[str, Any]:
    plan = build_viewpoint_inspection_plan(
        scenario,
        safety_aware=True,
        inspection_radius=inspection_radius,
    )
    record = evaluate_plan(scenario, plan)

    feasible = (
        plan.complete
        and float(record["success"]) == 1.0
        and float(record["hazard_recall"]) == 1.0
        and float(record["safety_cost"]) <= 1e-12
    )

    difficulty_score = (
        float(plan.movement_actions)
        + 2.0 * float(plan.inspection_actions)
        + 0.25 * len(scenario.obstacles)
        + 1.5 * len(scenario.workers)
        + 0.75 * len(scenario.restricted_zones)
    )

    return {
        "scenario_id": scenario.scenario_id,
        "source_split": scenario.split,
        "feasible": feasible,
        "tier": None,
        "difficulty_score": round(difficulty_score, 6),
        "plan_actions": len(plan.actions),
        "movement_actions": plan.movement_actions,
        "inspection_actions": plan.inspection_actions,
        "hazard_count": len(scenario.hazards),
        "obstacle_count": len(scenario.obstacles),
        "worker_count": len(scenario.workers),
        "restricted_count": len(scenario.restricted_zones),
        "hazard_recall": float(record["hazard_recall"]),
        "safety_cost": float(record["safety_cost"]),
        "success": float(record["success"]),
    }


def assign_training_tiers(
    entries: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    assigned = [dict(entry) for entry in entries]
    feasible_entries = sorted(
        (entry for entry in assigned if bool(entry["feasible"])),
        key=lambda entry: (
            float(entry["difficulty_score"]),
            int(entry["plan_actions"]),
            str(entry["scenario_id"]),
        ),
    )

    if not feasible_entries:
        raise ValueError("No safe-feasible scenarios were found.")

    feasible_count = len(feasible_entries)

    for index, entry in enumerate(feasible_entries):
        tier_index = min(
            len(CURRICULUM_TIERS) - 1,
            index * len(CURRICULUM_TIERS) // feasible_count,
        )
        entry["tier"] = CURRICULUM_TIERS[tier_index]

    return assigned


def _build_split_manifest(
    scenarios: Sequence[Scenario],
    *,
    inspection_radius: int,
    assign_tiers: bool,
) -> dict[str, Any]:
    if not scenarios:
        raise ValueError("At least one scenario is required.")

    entries = [
        evaluate_curriculum_scenario(
            scenario,
            inspection_radius=inspection_radius,
        )
        for scenario in scenarios
    ]

    if assign_tiers:
        entries = assign_training_tiers(entries)

    feasible_entries = [entry for entry in entries if bool(entry["feasible"])]
    tier_counts = {
        tier: sum(entry.get("tier") == tier for entry in feasible_entries)
        for tier in CURRICULUM_TIERS
    }

    return {
        "source_split": scenarios[0].split,
        "total_count": len(entries),
        "feasible_count": len(feasible_entries),
        "excluded_count": len(entries) - len(feasible_entries),
        "tier_counts": tier_counts,
        "entries": sorted(entries, key=lambda entry: str(entry["scenario_id"])),
    }


def build_curriculum_manifest(
    train_scenarios: Sequence[Scenario],
    validation_scenarios: Sequence[Scenario],
    *,
    source_manifest_path: str | Path,
    inspection_radius: int = 2,
) -> dict[str, Any]:
    source_manifest = Path(source_manifest_path)

    if inspection_radius < 1:
        raise ValueError("inspection_radius must be at least one.")
    if not source_manifest.is_file():
        raise FileNotFoundError(f"Source scenario manifest was not found: {source_manifest}")

    return {
        "schema_version": CURRICULUM_SCHEMA_VERSION,
        "planner": "safe_viewpoint_astar",
        "inspection_radius": inspection_radius,
        "selection": {
            "require_complete_plan": True,
            "required_hazard_recall": 1.0,
            "maximum_safety_cost": 0.0,
        },
        "difficulty_score": {
            "movement_action_weight": 1.0,
            "inspection_action_weight": 2.0,
            "obstacle_weight": 0.25,
            "worker_weight": 1.5,
            "restricted_zone_weight": 0.75,
            "tier_assignment": "balanced_rank_thirds",
        },
        "source_manifest": {
            "path": source_manifest.as_posix(),
            "sha256": sha256_file(source_manifest),
        },
        "splits": {
            "train": _build_split_manifest(
                train_scenarios,
                inspection_radius=inspection_radius,
                assign_tiers=True,
            ),
            "validation": _build_split_manifest(
                validation_scenarios,
                inspection_radius=inspection_radius,
                assign_tiers=False,
            ),
        },
    }


def save_curriculum_manifest(
    manifest: dict[str, Any],
    output_path: str | Path,
) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return path


def load_curriculum_manifest(
    manifest_path: str | Path,
    *,
    source_manifest_path: str | Path | None = None,
    verify_source_hash: bool = True,
) -> dict[str, Any]:
    path = Path(manifest_path)

    if not path.is_file():
        raise FileNotFoundError(f"Curriculum manifest was not found: {path}")

    manifest = json.loads(path.read_text(encoding="utf-8"))

    if int(manifest.get("schema_version", -1)) != CURRICULUM_SCHEMA_VERSION:
        raise ValueError("Unsupported curriculum manifest schema version.")

    if verify_source_hash:
        source_path = (
            Path(source_manifest_path)
            if source_manifest_path is not None
            else Path(str(manifest["source_manifest"]["path"]))
        )
        actual_hash = sha256_file(source_path)
        expected_hash = str(manifest["source_manifest"]["sha256"])

        if actual_hash != expected_hash:
            raise ValueError(
                "The scenario dataset manifest does not match the curriculum manifest."
            )

    return manifest


def _scenario_index(
    scenarios: Sequence[Scenario],
) -> dict[str, Scenario]:
    index = {scenario.scenario_id: scenario for scenario in scenarios}

    if len(index) != len(scenarios):
        raise ValueError("Scenario identifiers must be unique.")

    return index


def scenario_tiers_from_manifest(
    scenarios: Sequence[Scenario],
    manifest: dict[str, Any],
) -> dict[str, tuple[Scenario, ...]]:
    index = _scenario_index(scenarios)
    split_manifest = manifest["splits"]["train"]
    tiers: dict[str, list[Scenario]] = {tier: [] for tier in CURRICULUM_TIERS}

    for entry in split_manifest["entries"]:
        if not bool(entry["feasible"]):
            continue

        scenario_id = str(entry["scenario_id"])
        tier = str(entry["tier"])

        if tier not in tiers:
            raise ValueError(f"Unknown curriculum tier: {tier}")
        if scenario_id not in index:
            raise KeyError(f"Scenario from curriculum manifest was not found: {scenario_id}")

        tiers[tier].append(index[scenario_id])

    resolved = {
        tier: tuple(sorted(values, key=lambda scenario: scenario.scenario_id))
        for tier, values in tiers.items()
    }

    if any(not values for values in resolved.values()):
        raise ValueError("Every curriculum tier must contain at least one scenario.")

    return resolved


def feasible_validation_scenarios_from_manifest(
    scenarios: Sequence[Scenario],
    manifest: dict[str, Any],
) -> tuple[Scenario, ...]:
    index = _scenario_index(scenarios)
    entries = manifest["splits"]["validation"]["entries"]
    scenario_ids = sorted(str(entry["scenario_id"]) for entry in entries if bool(entry["feasible"]))

    missing = [scenario_id for scenario_id in scenario_ids if scenario_id not in index]
    if missing:
        raise KeyError(
            "Validation scenarios from the curriculum manifest were not found: "
            + ", ".join(missing[:5])
        )

    selected = tuple(index[scenario_id] for scenario_id in scenario_ids)

    if not selected:
        raise ValueError("No safe-feasible validation scenarios were found.")

    return selected
