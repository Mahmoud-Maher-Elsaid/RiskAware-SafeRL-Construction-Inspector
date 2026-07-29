from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from dataclasses import fields
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from riskaware_saferrl.baselines import CausalObservationExpert
from riskaware_saferrl.envs import (
    GridEnvironmentConfig,
    ResearchConstructionEnvV2,
)

WORLDS = ("site_small", "site_medium", "site_dynamic")
PROFILES = {
    "low": (0.75, 0.75, 0.0),
    "medium": (1.0, 1.0, 0.1),
    "high": (1.5, 1.5, 0.2),
}
SEED_RANGES = {
    "train": (0, 100),
    "validation": (1000, 30),
    "test": (2000, 30),
}


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


def run_episode(
    world: str,
    profile: str,
    seed: int,
    split: str,
) -> dict[str, Any]:
    hazard_multiplier, worker_multiplier, noise = PROFILES[profile]
    environment = ResearchConstructionEnvV2(
        load_config(world, hazard_multiplier, worker_multiplier, noise)
    )
    observation, _ = environment.reset(seed=seed)
    expert = CausalObservationExpert()
    invalid_actions = 0
    shield_interventions = 0
    proposed_counts = np.zeros(5, dtype=np.int64)
    executed_counts = np.zeros(5, dtype=np.int64)
    while True:
        decision = expert.decide(observation)
        proposed_counts[decision.proposed_action] += 1
        executed_counts[decision.final_action] += 1
        invalid_actions += int(not bool(observation["action_mask"][decision.final_action]))
        shield_interventions += int(decision.shield_decision != "accept")
        observation, _, terminated, truncated, info = environment.step(decision.final_action)
        if terminated or truncated:
            break
    constraint_violations = (
        int(environment.collisions)
        + int(environment.near_misses)
        + int(environment.restricted_violations)
    )
    return {
        "split": split,
        "world": world,
        "profile": profile,
        "seed": seed,
        "hazard_multiplier": hazard_multiplier,
        "worker_multiplier": worker_multiplier,
        "perception_false_negative_rate": noise,
        "success": bool(info["success"]),
        "hazard_recall": float(info["hazard_recall"]),
        "inspection_coverage": float(info["inspection_coverage"]),
        "explored_area_coverage": float(info["explored_area_coverage"]),
        "collisions": int(environment.collisions),
        "collision_rate": float(environment.collisions / max(1, environment.steps)),
        "safety_cost": float(environment.cumulative_cost),
        "constraint_violations": constraint_violations,
        "near_misses": int(environment.near_misses),
        "restricted_violations": int(environment.restricted_violations),
        "path_length": float(environment.path_length),
        "steps": int(environment.steps),
        "invalid_actions": invalid_actions,
        "shield_interventions": shield_interventions,
        "proposed_action_counts": proposed_counts.tolist(),
        "executed_action_counts": executed_counts.tolist(),
    }


def aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    proposed = np.sum(
        np.asarray([row["proposed_action_counts"] for row in rows], dtype=np.int64),
        axis=0,
    )
    executed = np.sum(
        np.asarray([row["executed_action_counts"] for row in rows], dtype=np.int64),
        axis=0,
    )
    return {
        "episodes": len(rows),
        "success_rate": float(np.mean([row["success"] for row in rows])),
        "hazard_recall": float(np.mean([row["hazard_recall"] for row in rows])),
        "inspection_coverage": float(np.mean([row["inspection_coverage"] for row in rows])),
        "explored_area_coverage": float(np.mean([row["explored_area_coverage"] for row in rows])),
        "collision_rate": float(
            sum(row["collisions"] for row in rows) / max(1, sum(row["steps"] for row in rows))
        ),
        "mean_safety_cost": float(np.mean([row["safety_cost"] for row in rows])),
        "constraint_violations": int(sum(row["constraint_violations"] for row in rows)),
        "mean_path_length": float(np.mean([row["path_length"] for row in rows])),
        "invalid_actions": int(sum(row["invalid_actions"] for row in rows)),
        "shield_interventions": int(sum(row["shield_interventions"] for row in rows)),
        "proposed_action_counts": proposed.tolist(),
        "executed_action_counts": executed.tolist(),
    }


def evaluate(output_dir: Path, seed_ranges: dict[str, tuple[int, int]]) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for split, (start, count) in seed_ranges.items():
        for seed in range(start, start + count):
            for world in WORLDS:
                for profile in PROFILES:
                    rows.append(run_episode(world, profile, seed, split))
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "evaluation.csv"
    scalar_fields = [key for key, value in rows[0].items() if not isinstance(value, list)]
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=scalar_fields)
        writer.writeheader()
        writer.writerows({key: row[key] for key in scalar_fields} for row in rows)

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[f"split:{row['split']}"].append(row)
        grouped[f"world:{row['world']}"].append(row)
        grouped[f"profile:{row['profile']}"].append(row)
    metrics = {name: aggregate(group) for name, group in sorted(grouped.items())}
    overall = aggregate(rows)
    world_success = {world: metrics[f"world:{world}"]["success_rate"] for world in WORLDS}
    passed = (
        world_success["site_small"] >= 0.90
        and world_success["site_medium"] >= 0.80
        and world_success["site_dynamic"] >= 0.65
        and overall["hazard_recall"] >= 0.80
        and overall["inspection_coverage"] >= 0.80
        and overall["invalid_actions"] == 0
        and overall["collision_rate"] <= 0.02
    )
    summary = {
        "status": "PASSED" if passed else "FAILED",
        "observation_only_contract": True,
        "seed_ranges": {
            split: {"start": start, "count": count} for split, (start, count) in seed_ranges.items()
        },
        "worlds": list(WORLDS),
        "profiles": {
            name: {
                "hazard_multiplier": values[0],
                "worker_multiplier": values[1],
                "perception_false_negative_rate": values[2],
            }
            for name, values in PROFILES.items()
        },
        "overall": overall,
        "groups": metrics,
        "gates": {
            "small_world_success": {
                "required": 0.90,
                "observed": world_success["site_small"],
            },
            "medium_world_success": {
                "required": 0.80,
                "observed": world_success["site_medium"],
            },
            "dynamic_world_success": {
                "required": 0.65,
                "observed": world_success["site_dynamic"],
            },
            "hazard_recall": {
                "required": 0.80,
                "observed": overall["hazard_recall"],
            },
            "inspection_coverage": {
                "required": 0.80,
                "observed": overall["inspection_coverage"],
            },
            "invalid_actions": {
                "required": 0,
                "observed": overall["invalid_actions"],
            },
            "collision_rate": {
                "required_maximum": 0.02,
                "observed": overall["collision_rate"],
            },
        },
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("reports/strong_policy_upgrade/causal_expert"),
    )
    parser.add_argument("--train-seeds", type=int, default=100)
    parser.add_argument("--validation-seeds", type=int, default=30)
    parser.add_argument("--test-seeds", type=int, default=30)
    args = parser.parse_args()
    seed_ranges = {
        "train": (0, args.train_seeds),
        "validation": (1000, args.validation_seeds),
        "test": (2000, args.test_seeds),
    }
    summary = evaluate(args.output_dir, seed_ranges)
    print(f"CAUSAL_EXPERT_GATE={summary['status']}")
    if summary["status"] != "PASSED":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
