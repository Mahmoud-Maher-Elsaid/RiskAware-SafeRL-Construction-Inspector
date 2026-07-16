from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
from stable_baselines3 import PPO

from riskaware_saferrl.envs import ConstructionInspectionEnv
from riskaware_saferrl.safety import SafetyShield
from riskaware_saferrl.scenario_dataset import load_scenarios


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=Path("data/scenarios"),
    )
    parser.add_argument("--split", type=str, default="validation")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--shield", action="store_true")
    parser.add_argument("--output-name", type=str, default=None)
    return parser.parse_args()


def mean(values: list[float]) -> float:
    return float(np.mean(np.asarray(values, dtype=np.float64)))


def main() -> None:
    args = parse_args()
    model = PPO.load(args.model)

    scenarios = load_scenarios(
        args.dataset_dir,
        args.split,
        verify_hash=True,
    )

    if args.limit is not None:
        if args.limit < 1:
            raise ValueError("--limit must be positive.")
        scenarios = scenarios[: args.limit]

    records: list[dict[str, Any]] = []

    for scenario in scenarios:
        environment = ConstructionInspectionEnv(scenario=scenario)

        if args.shield:
            environment = SafetyShield(environment)

        observation, _ = environment.reset(seed=0)

        total_reward = 0.0
        total_cost = 0.0
        collision_cost = 0.0
        worker_cost = 0.0
        restricted_cost = 0.0
        shield_interventions = 0
        steps = 0

        while True:
            action, _ = model.predict(
                observation,
                deterministic=True,
            )

            observation, reward, terminated, truncated, info = environment.step(int(action))

            total_reward += float(reward)
            total_cost += float(info["cost"])
            collision_cost += float(info["cost_collision"])
            worker_cost += float(info["cost_worker"])
            restricted_cost += float(info["cost_restricted"])
            shield_interventions += int(info.get("shield_active", False))
            steps += 1

            if terminated or truncated:
                break

        records.append(
            {
                "scenario_id": scenario.scenario_id,
                "split": scenario.split,
                "reward": total_reward,
                "safety_cost": total_cost,
                "collision_cost": collision_cost,
                "worker_cost": worker_cost,
                "restricted_cost": restricted_cost,
                "hazard_recall": float(info["hazard_recall"]),
                "coverage": float(info["coverage"]),
                "success": bool(info["success"]),
                "steps": steps,
                "shield_interventions": shield_interventions,
            }
        )

        environment.close()

    output_directory = Path("artifacts/results")
    output_directory.mkdir(parents=True, exist_ok=True)

    output_name = args.output_name or f"{args.model.stem}_{args.split}"
    csv_path = output_directory / f"{output_name}_episodes.csv"
    json_path = output_directory / f"{output_name}_summary.json"

    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)

    summary = {
        "model": str(args.model),
        "split": args.split,
        "scenario_count": len(records),
        "shield": args.shield,
        "reward_mean": mean([float(record["reward"]) for record in records]),
        "safety_cost_mean": mean([float(record["safety_cost"]) for record in records]),
        "collision_cost_mean": mean([float(record["collision_cost"]) for record in records]),
        "worker_cost_mean": mean([float(record["worker_cost"]) for record in records]),
        "restricted_cost_mean": mean([float(record["restricted_cost"]) for record in records]),
        "hazard_recall_mean": mean([float(record["hazard_recall"]) for record in records]),
        "coverage_mean": mean([float(record["coverage"]) for record in records]),
        "success_rate": mean([float(bool(record["success"])) for record in records]),
        "shield_interventions_mean": mean(
            [float(record["shield_interventions"]) for record in records]
        ),
    }

    json_path.write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )

    print(json.dumps(summary, indent=2))
    print(f"Episode results: {csv_path}")
    print(f"Summary: {json_path}")


if __name__ == "__main__":
    main()
