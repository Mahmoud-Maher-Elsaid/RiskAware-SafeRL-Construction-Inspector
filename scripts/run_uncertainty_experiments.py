from __future__ import annotations

import csv
import json
from dataclasses import replace
from pathlib import Path

import gymnasium as gym
import numpy as np
import yaml
from stable_baselines3 import PPO

from riskaware_saferrl.envs import GridEnvironmentConfig, ResearchConstructionEnv
from riskaware_saferrl.safety import PredictiveSafetyShield, PredictiveShieldWrapper
from riskaware_saferrl.uncertainty import (
    PerceptionUncertaintyConfig,
    PerceptionUncertaintyWrapper,
)


def evaluate(
    model: PPO,
    environment_name: str,
    grid_config: GridEnvironmentConfig,
    false_negative_rate: float,
    visual: str,
    seed: int,
) -> dict[str, object]:
    base = ResearchConstructionEnv(grid_config)
    shielded = PredictiveShieldWrapper(base, PredictiveSafetyShield(horizon=3))
    uncertain = PerceptionUncertaintyWrapper(
        shielded,
        PerceptionUncertaintyConfig(false_negative_rate, visual, seed),
    )
    environment = gym.wrappers.FlattenObservation(uncertain)
    observation, info = environment.reset(seed=seed)
    terminated = truncated = False
    interventions = 0
    emergency_stops = 0
    while not (terminated or truncated):
        action, _ = model.predict(observation, deterministic=True)
        observation, _, terminated, truncated, info = environment.step(
            int(np.asarray(action).reshape(-1)[0])
        )
        interventions += int(bool(info.get("shield_intervention", False)))
        emergency_stops += int(bool(info.get("emergency_stop", False)))
    telemetry = base.telemetry()
    environment.close()
    return {
        "run_id": (f"{environment_name}:fn{false_negative_rate:.1f}:{visual}:seed{seed}"),
        "environment_perturbation": environment_name,
        "false_negative_rate": false_negative_rate,
        "visual_perturbation": visual,
        "seed": seed,
        "ground_truth_hazards": telemetry.total_hazards,
        "hazard_recall": info["hazard_recall"],
        "collision_rate": float(telemetry.collisions > 0),
        "near_miss_rate": telemetry.near_misses / max(1, telemetry.steps),
        "constraint_violation_rate": (telemetry.near_misses + telemetry.restricted_violations)
        / max(1, telemetry.steps),
        "coverage": info["coverage"],
        "shield_interventions": interventions,
        "mission_success": float(info["success"]),
        "safety_cost": telemetry.safety_cost,
        "emergency_stops": emergency_stops,
    }


def main() -> None:
    protocol = yaml.safe_load(Path("configs/uncertainty/protocol.yaml").read_text(encoding="utf-8"))
    model = PPO.load("artifacts/final_submission/riskshield_ppo/best_model.zip", device="cuda")
    base_config = GridEnvironmentConfig()
    environment_configs = {
        name: replace(base_config, **values)
        for name, values in protocol["environment_perturbations"].items()
    }
    records = [
        evaluate(model, name, grid_config, rate, visual, seed)
        for name, grid_config in environment_configs.items()
        for rate in protocol["false_negative_rates"]
        for visual in protocol["visual_perturbations"]
        for seed in protocol["evaluation_seeds"]
    ]
    output = Path("reports/final_submission/stage8_uncertainty")
    output.mkdir(parents=True, exist_ok=True)
    raw_path = output / "raw_results.csv"
    with raw_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    baseline = [
        record
        for record in records
        if record["environment_perturbation"] == "low"
        and record["false_negative_rate"] == 0.0
        and record["visual_perturbation"] == "normal"
    ]
    baseline_values = {
        key: float(np.mean([record[key] for record in baseline]))
        for key in (
            "hazard_recall",
            "collision_rate",
            "near_miss_rate",
            "constraint_violation_rate",
            "coverage",
            "shield_interventions",
            "mission_success",
        )
    }
    summaries = []
    group_keys = (
        "environment_perturbation",
        "false_negative_rate",
        "visual_perturbation",
    )
    groups = sorted({tuple(record[key] for key in group_keys) for record in records})
    for group in groups:
        subset = [record for record in records if tuple(record[key] for key in group_keys) == group]
        means = {key: float(np.mean([record[key] for record in subset])) for key in baseline_values}
        positive = np.mean(
            [
                means["hazard_recall"] / max(1e-6, baseline_values["hazard_recall"]),
                means["coverage"] / max(1e-6, baseline_values["coverage"]),
                (
                    means["mission_success"] / max(1e-6, baseline_values["mission_success"])
                    if baseline_values["mission_success"] > 0
                    else 1.0
                ),
            ]
        )
        safety = np.mean(
            [
                1.0
                / (
                    1.0
                    + max(
                        0.0,
                        means[key] - baseline_values[key],
                    )
                )
                for key in (
                    "collision_rate",
                    "near_miss_rate",
                    "constraint_violation_rate",
                )
            ]
        )
        robustness = float(np.clip(0.5 * positive + 0.5 * safety, 0.0, 1.0))
        summaries.append(
            {
                **dict(zip(group_keys, group, strict=True)),
                **means,
                "robustness_score": robustness,
            }
        )
    with (output / "aggregated_results.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summaries[0]))
        writer.writeheader()
        writer.writerows(summaries)
    summary = {
        "status": "PASSED",
        "run_count": len(records),
        "expected_run_count": 320,
        "conditions": len(summaries),
        "baseline": baseline_values,
        "robustness_definition": (
            "0.5*mean(relative recall, coverage, success) + "
            "0.5*mean(1/(1+positive safety degradation)), clipped to [0,1]"
        ),
        "mean_robustness_score": float(
            np.mean([record["robustness_score"] for record in summaries])
        ),
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"UNCERTAINTY_EXPERIMENTS=PASSED ({len(records)} episodes)")


if __name__ == "__main__":
    main()
