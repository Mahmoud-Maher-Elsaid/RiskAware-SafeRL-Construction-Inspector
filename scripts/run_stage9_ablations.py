from __future__ import annotations

import argparse
import csv
import json
from dataclasses import replace
from pathlib import Path

import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO

from riskaware_saferrl.envs import ResearchConstructionEnv
from riskaware_saferrl.safety import PredictiveSafetyShield, PredictiveShieldWrapper
from riskaware_saferrl.training import load_grid_config

OUTPUT = Path("reports/final_submission/stage9_benchmark")
SCENARIOS = (
    Path("configs/grid/site_small.yaml"),
    Path("configs/grid/site_medium.yaml"),
    Path("configs/grid/site_dynamic.yaml"),
)
VARIANTS = (
    {
        "name": "riskshield_ppo_full",
        "checkpoint": "riskshield",
        "shield_horizon": 3,
        "cv_risk_injection": True,
        "action_masking": True,
        "constrained_cost_objective": True,
        "difference": "Reference: constrained checkpoint, k=3 shield, semantic risk observations, action-mask projection.",
    },
    {
        "name": "without_predictive_safety_shield",
        "checkpoint": "riskshield",
        "shield_horizon": 0,
        "cv_risk_injection": True,
        "action_masking": True,
        "constrained_cost_objective": True,
        "difference": "Predictive Safety Shield disabled; all other reference mechanisms retained.",
    },
    {
        "name": "without_cv_risk_injection",
        "checkpoint": "riskshield",
        "shield_horizon": 3,
        "cv_risk_injection": False,
        "action_masking": True,
        "constrained_cost_objective": True,
        "difference": "Agent perception false-negative rate set to 1.0; simulator truth remains available to metrics and shield.",
    },
    {
        "name": "without_action_masking",
        "checkpoint": "riskshield",
        "shield_horizon": 3,
        "cv_risk_injection": True,
        "action_masking": False,
        "constrained_cost_objective": True,
        "difference": "Invalid-action projection disabled; predictive shield and semantic risk retained.",
    },
    {
        "name": "without_constrained_cost_objective",
        "checkpoint": "ppo",
        "shield_horizon": 3,
        "cv_risk_injection": True,
        "action_masking": True,
        "constrained_cost_objective": False,
        "difference": "Ordinary PPO checkpoint replaces RiskShield-PPO; inference shield, risk observations, and mask retained.",
    },
    {
        "name": "one_step_shield",
        "checkpoint": "riskshield",
        "shield_horizon": 1,
        "cv_risk_injection": True,
        "action_masking": True,
        "constrained_cost_objective": True,
        "difference": "Predictive shield horizon reduced from k=3 to k=1.",
    },
)


def evaluate(
    variant: dict[str, object],
    model: PPO,
    scenario: Path,
    seed: int,
) -> dict[str, object]:
    config = load_grid_config(scenario)
    if not variant["cv_risk_injection"]:
        config = replace(config, perception_false_negative_rate=1.0)
    base = ResearchConstructionEnv(config)
    horizon = int(variant["shield_horizon"])
    environment: gym.Env = base
    if horizon:
        environment = PredictiveShieldWrapper(base, PredictiveSafetyShield(horizon=horizon))
    environment = gym.wrappers.FlattenObservation(environment)
    observation, info = environment.reset(seed=seed)
    terminated = truncated = False
    interventions = emergency_stops = mask_projections = 0
    episode_return = 0.0
    while not (terminated or truncated):
        proposed, _ = model.predict(observation, deterministic=True)
        action = int(np.asarray(proposed).reshape(-1)[0])
        if variant["action_masking"] and not bool(base.action_masks()[action]):
            valid = np.flatnonzero(base.action_masks())
            action = int(valid[0])
            mask_projections += 1
        observation, reward, terminated, truncated, info = environment.step(action)
        episode_return += float(reward)
        interventions += int(bool(info.get("shield_intervention", False)))
        emergency_stops += int(bool(info.get("emergency_stop", False)))
    telemetry = base.telemetry()
    environment.close()
    return {
        "run_id": f"{variant['name']}:{scenario.stem}:seed{seed}",
        "variant": variant["name"],
        "exact_configuration_difference": variant["difference"],
        "scenario": scenario.stem,
        "evaluation_seed": seed,
        "checkpoint": variant["checkpoint"],
        "shield_horizon": horizon,
        "cv_risk_injection": bool(variant["cv_risk_injection"]),
        "action_masking": bool(variant["action_masking"]),
        "constrained_cost_objective": bool(variant["constrained_cost_objective"]),
        "episode_return": episode_return,
        "hazard_recall": float(info["hazard_recall"]),
        "inspection_coverage": float(info["coverage"]),
        "collision_count": telemetry.collisions,
        "near_miss_count": telemetry.near_misses,
        "constraint_violations": (telemetry.near_misses + telemetry.restricted_violations),
        "safety_cost": telemetry.safety_cost,
        "success": bool(info["success"]),
        "path_length": telemetry.path_length,
        "energy_usage": telemetry.energy_usage,
        "shield_interventions": interventions,
        "emergency_stops": emergency_stops,
        "mask_projections": mask_projections,
        "episode_steps": telemetry.steps,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    existing_results = OUTPUT / "ablation_results.csv"
    existing_summary = OUTPUT / "ablation_summary.json"
    if not args.force and existing_results.is_file() and existing_summary.is_file():
        records = list(csv.DictReader(existing_results.open(encoding="utf-8")))
        summary = json.loads(existing_summary.read_text(encoding="utf-8"))
        if (
            len(records) == 180
            and len({record["run_id"] for record in records}) == 180
            and summary.get("status") == "PASSED"
        ):
            print("STAGE9_ABLATIONS=PASSED (180 cached runs)")
            return
    models = {
        "ppo": PPO.load("artifacts/final_submission/ppo/best_model.zip", device="cuda"),
        "riskshield": PPO.load(
            "artifacts/final_submission/riskshield_ppo/best_model.zip",
            device="cuda",
        ),
    }
    records = [
        evaluate(variant, models[str(variant["checkpoint"])], scenario, seed)
        for variant in VARIANTS
        for scenario in SCENARIOS
        for seed in range(10)
    ]
    OUTPUT.mkdir(parents=True, exist_ok=True)
    with (OUTPUT / "ablation_results.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    summary = {
        "status": "PASSED",
        "run_count": len(records),
        "unique_run_count": len({record["run_id"] for record in records}),
        "evaluation_design": "Six variants × three scenarios × ten paired evaluation seeds.",
        "variants": {},
    }
    for variant in VARIANTS:
        subset = [record for record in records if record["variant"] == variant["name"]]
        summary["variants"][str(variant["name"])] = {
            **variant,
            "episodes": len(subset),
            "mean_hazard_recall": float(np.mean([record["hazard_recall"] for record in subset])),
            "mean_coverage": float(np.mean([record["inspection_coverage"] for record in subset])),
            "mean_safety_cost": float(np.mean([record["safety_cost"] for record in subset])),
            "collision_rate": float(np.mean([record["collision_count"] > 0 for record in subset])),
            "success_rate": float(np.mean([record["success"] for record in subset])),
            "mean_shield_interventions": float(
                np.mean([record["shield_interventions"] for record in subset])
            ),
            "mean_mask_projections": float(
                np.mean([record["mask_projections"] for record in subset])
            ),
        }
    (OUTPUT / "ablation_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    (OUTPUT / "ablation_manifest.json").write_text(
        json.dumps(
            {
                "variants": list(VARIANTS),
                "scenarios": [path.as_posix() for path in SCENARIOS],
                "evaluation_seeds": list(range(10)),
                "expected_runs": 180,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"STAGE9_ABLATIONS=PASSED ({len(records)} runs)")


if __name__ == "__main__":
    main()
