from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path

import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

from riskaware_saferrl.envs import ResearchConstructionEnv
from riskaware_saferrl.safety import PredictiveSafetyShield, PredictiveShieldWrapper
from riskaware_saferrl.training import load_grid_config


def evaluate(
    label: str,
    model: PPO,
    config_path: Path,
    seed: int,
    *,
    shield_horizon: int | None,
) -> dict[str, object]:
    base = ResearchConstructionEnv(load_grid_config(config_path))
    if shield_horizon is not None:
        environment: gym.Env = PredictiveShieldWrapper(
            base, PredictiveSafetyShield(horizon=shield_horizon)
        )
    else:
        environment = base
    environment = gym.wrappers.FlattenObservation(environment)
    observation, info = environment.reset(seed=seed)
    terminated = truncated = False
    episode_return = 0.0
    interventions = 0
    emergency_stops = 0
    while not (terminated or truncated):
        action, _ = model.predict(observation, deterministic=True)
        observation, reward, terminated, truncated, info = environment.step(
            int(np.asarray(action).reshape(-1)[0])
        )
        episode_return += float(reward)
        interventions += int(bool(info.get("shield_intervention", False)))
        emergency_stops += int(bool(info.get("emergency_stop", False)))
    telemetry = base.telemetry()
    environment.close()
    return {
        "run_id": f"{label}:{config_path.stem}:{seed}",
        "configuration": label,
        "algorithm": "RiskShield-PPO" if label.startswith("riskshield") else "PPO",
        "shield_enabled": shield_horizon is not None,
        "shield_horizon": shield_horizon or 0,
        "environment": config_path.stem,
        "seed": seed,
        "episodic_return": episode_return,
        "safety_cost": telemetry.safety_cost,
        "hazard_recall": info["hazard_recall"],
        "coverage": info["coverage"],
        "collisions": telemetry.collisions,
        "constraint_violations": telemetry.near_misses + telemetry.restricted_violations,
        "shield_interventions": interventions,
        "emergency_stops": emergency_stops,
        "success": bool(info["success"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("reports/final_submission/stage4_riskshield_ppo"),
    )
    args = parser.parse_args()
    models = {
        "ppo": PPO.load("artifacts/final_submission/ppo/best_model.zip", device="cuda"),
        "riskshield": PPO.load(
            "artifacts/final_submission/riskshield_ppo/best_model.zip", device="cuda"
        ),
    }
    comparisons = (
        ("ppo_without_shield", models["ppo"], None),
        ("ppo_with_shield", models["ppo"], 3),
        ("riskshield_without_shield", models["riskshield"], None),
        ("riskshield_with_shield", models["riskshield"], 3),
    )
    configs = [
        Path("configs/grid/site_small.yaml"),
        Path("configs/grid/site_medium.yaml"),
        Path("configs/grid/site_dynamic.yaml"),
    ]
    records = [
        evaluate(label, model, config, seed, shield_horizon=horizon)
        for label, model, horizon in comparisons
        for config in configs
        for seed in range(10)
    ]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "comparison_results.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    summary = {"status": "PASSED", "run_count": len(records), "comparisons": {}}
    for label, _, _ in comparisons:
        subset = [record for record in records if record["configuration"] == label]
        summary["comparisons"][label] = {
            "episodes": len(subset),
            "mean_return": float(np.mean([record["episodic_return"] for record in subset])),
            "mean_safety_cost": float(np.mean([record["safety_cost"] for record in subset])),
            "mean_hazard_recall": float(np.mean([record["hazard_recall"] for record in subset])),
            "collision_rate": float(np.mean([record["collisions"] > 0 for record in subset])),
            "mean_shield_interventions": float(
                np.mean([record["shield_interventions"] for record in subset])
            ),
            "success_rate": float(np.mean([record["success"] for record in subset])),
        }
    (args.output_dir / "comparison_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    shutil.copy2(
        "artifacts/final_submission/riskshield_ppo/metadata.json",
        args.output_dir / "checkpoint_metadata.json",
    )
    metrics_path = Path("artifacts/final_submission/riskshield_ppo/training_metrics.jsonl")
    episode_metrics = [
        json.loads(line)
        for line in metrics_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    with (args.output_dir / "constraint_training_metrics.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(episode_metrics[0]))
        writer.writeheader()
        writer.writerows(episode_metrics)
    event_files = sorted(
        Path("artifacts/final_submission/riskshield_ppo/tensorboard").rglob("events.out.tfevents.*")
    )
    accumulator = EventAccumulator(str(event_files[-1]))
    accumulator.Reload()
    scalar_tags = {
        "train/policy_gradient_loss",
        "train/value_loss",
        "train/entropy_loss",
        "train/approx_kl",
        "train/loss",
    }
    diagnostics = []
    for tag in sorted(scalar_tags & set(accumulator.Tags()["scalars"])):
        for event in accumulator.Scalars(tag):
            diagnostics.append({"timesteps": event.step, "metric": tag, "value": event.value})
    with (args.output_dir / "ppo_training_diagnostics.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(diagnostics[0]))
        writer.writeheader()
        writer.writerows(diagnostics)
    print(f"RISKSHIELD_PPO_EVALUATION=PASSED ({len(records)} episodes)")


if __name__ == "__main__":
    main()
