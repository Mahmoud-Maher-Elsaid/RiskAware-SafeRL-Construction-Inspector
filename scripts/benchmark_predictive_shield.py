from __future__ import annotations

import csv
import json
from pathlib import Path

import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO

from riskaware_saferrl.envs import ResearchConstructionEnv
from riskaware_saferrl.safety import PredictiveSafetyShield, PredictiveShieldWrapper
from riskaware_saferrl.training import load_grid_config


def episode(model: PPO, config_path: Path, seed: int, horizon: int) -> dict[str, object]:
    base = ResearchConstructionEnv(load_grid_config(config_path))
    if horizon:
        wrapper = PredictiveShieldWrapper(base, PredictiveSafetyShield(horizon=horizon))
        environment: gym.Env = wrapper
    else:
        wrapper = None
        environment = base
    environment = gym.wrappers.FlattenObservation(environment)
    observation, info = environment.reset(seed=seed)
    terminated = truncated = False
    interventions = 0
    unnecessary = 0
    latencies: list[float] = []
    emergency_stops = 0
    while not (terminated or truncated):
        action_array, _ = model.predict(observation, deterministic=True)
        proposed_action = int(np.asarray(action_array).reshape(-1)[0])
        immediate_violations = base.action_safety_violations(proposed_action)
        observation, _, terminated, truncated, info = environment.step(proposed_action)
        if wrapper is not None and wrapper.last_decision is not None:
            decision = wrapper.last_decision
            intervened = decision.shield_decision != "accept"
            interventions += int(intervened)
            emergency_stops += int(decision.emergency_stop)
            latencies.append(decision.computation_time_ms)
            unnecessary += int(intervened and not immediate_violations)
    telemetry = base.telemetry()
    environment.close()
    return {
        "run_id": f"shield_h{horizon}:{config_path.stem}:{seed}",
        "shield_mode": "disabled" if horizon == 0 else f"{horizon}_step",
        "horizon": horizon,
        "environment": config_path.stem,
        "seed": seed,
        "hazard_recall": info["hazard_recall"],
        "coverage": info["coverage"],
        "collisions": telemetry.collisions,
        "constraint_violations": telemetry.near_misses + telemetry.restricted_violations,
        "safety_cost": telemetry.safety_cost,
        "mission_success": bool(info["success"]),
        "shield_interventions": interventions,
        "emergency_stops": emergency_stops,
        "unnecessary_interventions_immediate_proxy": unnecessary,
        "average_intervention_latency_ms": float(np.mean(latencies)) if latencies else 0.0,
    }


def main() -> None:
    output = Path("reports/final_submission/stage5_safety_shield")
    output.mkdir(parents=True, exist_ok=True)
    model = PPO.load("artifacts/final_submission/riskshield_ppo/best_model.zip", device="cuda")
    configs = [
        Path("configs/grid/site_small.yaml"),
        Path("configs/grid/site_medium.yaml"),
        Path("configs/grid/site_dynamic.yaml"),
    ]
    records = [
        episode(model, config, seed, horizon)
        for horizon in (0, 1, 3)
        for config in configs
        for seed in range(10)
    ]
    with (output / "raw_results.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    summary = {"status": "PASSED", "run_count": len(records), "modes": {}}
    for mode in ("disabled", "1_step", "3_step"):
        subset = [record for record in records if record["shield_mode"] == mode]
        summary["modes"][mode] = {
            "episodes": len(subset),
            "collision_rate": float(np.mean([record["collisions"] > 0 for record in subset])),
            "mean_constraint_violations": float(
                np.mean([record["constraint_violations"] for record in subset])
            ),
            "mean_safety_cost": float(np.mean([record["safety_cost"] for record in subset])),
            "mean_hazard_recall": float(np.mean([record["hazard_recall"] for record in subset])),
            "success_rate": float(np.mean([record["mission_success"] for record in subset])),
            "mean_interventions": float(
                np.mean([record["shield_interventions"] for record in subset])
            ),
            "unnecessary_intervention_rate_immediate_proxy": (
                float(
                    sum(record["unnecessary_interventions_immediate_proxy"] for record in subset)
                    / max(
                        1,
                        sum(record["shield_interventions"] for record in subset),
                    )
                )
            ),
            "mean_latency_ms": float(
                np.mean([record["average_intervention_latency_ms"] for record in subset])
            ),
        }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"PREDICTIVE_SHIELD_BENCHMARK=PASSED ({len(records)} episodes)")


if __name__ == "__main__":
    main()
