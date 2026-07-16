from __future__ import annotations

import csv
import json
import math
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from riskaware_saferrl.envs import ConstructionInspectionEnv
from riskaware_saferrl.safety import SafetyShield
from riskaware_saferrl.scenarios import Scenario


class PredictivePolicy(Protocol):
    def predict(
        self,
        observation: dict[str, np.ndarray],
        deterministic: bool = True,
    ) -> tuple[Any, Any]: ...


METRIC_NAMES = (
    "reward",
    "safety_cost",
    "collision_cost",
    "worker_cost",
    "restricted_cost",
    "hazard_recall",
    "coverage",
    "success",
    "steps",
    "shield_interventions",
)


def summarize_values(values: Sequence[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)

    if array.size == 0:
        raise ValueError("Cannot summarize an empty metric sequence.")

    mean = float(np.mean(array))
    std = float(np.std(array, ddof=1)) if array.size > 1 else 0.0
    ci95 = float(1.96 * std / math.sqrt(array.size)) if array.size > 1 else 0.0

    return {
        "mean": mean,
        "std": std,
        "ci95": ci95,
        "min": float(np.min(array)),
        "max": float(np.max(array)),
    }


def evaluate_policy_on_scenarios(
    model: PredictivePolicy,
    scenarios: Sequence[Scenario],
    *,
    use_shield: bool = False,
    deterministic: bool = True,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if not scenarios:
        raise ValueError("At least one scenario is required for evaluation.")

    records: list[dict[str, Any]] = []

    for scenario in scenarios:
        environment = ConstructionInspectionEnv(scenario=scenario)

        if use_shield:
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
                deterministic=deterministic,
            )

            observation, reward, terminated, truncated, info = environment.step(
                int(np.asarray(action).item())
            )

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
                "success": float(bool(info["success"])),
                "steps": float(steps),
                "shield_interventions": float(shield_interventions),
            }
        )

        environment.close()

    summary = {
        "scenario_count": len(records),
        "split": scenarios[0].split,
        "shield": use_shield,
        "deterministic": deterministic,
        "metrics": {
            metric_name: summarize_values([float(record[metric_name]) for record in records])
            for metric_name in METRIC_NAMES
        },
    }

    return records, summary


def compute_selection_score(
    summary: dict[str, Any],
    *,
    selection_metric: str,
    safety_cost_limit: float,
) -> float:
    metrics = summary["metrics"]

    if selection_metric == "reward":
        return float(metrics["reward"]["mean"])

    if selection_metric == "hazard_recall":
        return float(metrics["hazard_recall"]["mean"])

    if selection_metric == "safe_hazard_recall":
        mean_cost = float(metrics["safety_cost"]["mean"])
        mean_recall = float(metrics["hazard_recall"]["mean"])

        if mean_cost <= safety_cost_limit:
            return mean_recall

        return -1.0 - (mean_cost - safety_cost_limit)

    raise ValueError(f"Unsupported selection metric: {selection_metric}")


def save_evaluation_results(
    records: Sequence[dict[str, Any]],
    summary: dict[str, Any],
    output_directory: str | Path,
    output_name: str,
) -> tuple[Path, Path]:
    output_path = Path(output_directory)
    output_path.mkdir(parents=True, exist_ok=True)

    csv_path = output_path / f"{output_name}_episodes.csv"
    json_path = output_path / f"{output_name}_summary.json"

    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)

    json_path.write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )

    return csv_path, json_path
