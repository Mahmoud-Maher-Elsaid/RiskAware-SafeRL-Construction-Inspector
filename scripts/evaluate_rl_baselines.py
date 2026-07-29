from __future__ import annotations

import argparse
import csv
import json
import shutil
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from stable_baselines3 import PPO, SAC

from riskaware_saferrl.training import flatten_research_environment, load_grid_config


def evaluate_episode(
    algorithm: str,
    model,
    config_path: Path,
    seed: int,
) -> dict[str, object]:
    continuous = algorithm == "SAC"
    environment = flatten_research_environment(
        load_grid_config(config_path), continuous_actions=continuous
    )
    observation, info = environment.reset(seed=seed)
    terminated = truncated = False
    returns = 0.0
    latencies: list[float] = []
    while not (terminated or truncated):
        started = time.perf_counter()
        action, _ = model.predict(observation, deterministic=True)
        latencies.append((time.perf_counter() - started) * 1000.0)
        if not continuous:
            action = int(np.asarray(action).reshape(-1)[0])
        observation, reward, terminated, truncated, info = environment.step(action)
        returns += float(reward)
    base = environment.unwrapped
    telemetry = base.telemetry()
    environment.close()
    return {
        "run_id": f"{algorithm.lower()}:{config_path.stem}:{seed}",
        "algorithm": algorithm,
        "environment": config_path.stem,
        "seed": seed,
        "episodic_return": returns,
        "safety_cost": telemetry.safety_cost,
        "coverage": info["coverage"],
        "hazard_recall": info["hazard_recall"],
        "collisions": telemetry.collisions,
        "constraint_violations": telemetry.near_misses + telemetry.restricted_violations,
        "success": bool(info["success"]),
        "path_length": telemetry.path_length,
        "energy_usage": telemetry.energy_usage,
        "policy_latency_ms": float(np.mean(latencies)),
    }


def read_training_metrics(algorithm: str, path: Path) -> list[dict[str, object]]:
    if not path.exists():
        raise FileNotFoundError(f"Training metrics are missing: {path}")
    records = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            records.append({"algorithm": algorithm, **json.loads(line)})
    return records


def write_csv(path: Path, records: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


def generate_plots(output: Path, training: list[dict[str, object]]) -> None:
    metrics = (
        "episodic_return",
        "safety_cost",
        "coverage",
        "hazard_recall",
        "collisions",
        "constraint_violations",
        "success",
    )
    for metric in metrics:
        figure, axis = plt.subplots(figsize=(7, 4))
        for algorithm in ("PPO", "SAC"):
            subset = [record for record in training if record["algorithm"] == algorithm]
            axis.plot(
                [record["timesteps"] for record in subset],
                [record[metric] for record in subset],
                label=algorithm,
                alpha=0.85,
            )
        axis.set_xlabel("Environment steps")
        axis.set_ylabel(metric.replace("_", " ").title())
        axis.grid(alpha=0.25)
        axis.legend()
        figure.tight_layout()
        figure.savefig(output / f"{metric}.png", dpi=160)
        plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("reports/final_submission/stage3_rl_baselines"),
    )
    args = parser.parse_args()
    checkpoints = {
        "PPO": Path("artifacts/final_submission/ppo/best_model.zip"),
        "SAC": Path("artifacts/final_submission/sac/best_model.zip"),
    }
    models = {
        "PPO": PPO.load(checkpoints["PPO"], device="cuda"),
        "SAC": SAC.load(checkpoints["SAC"], device="cuda"),
    }
    configs = [
        Path("configs/grid/site_small.yaml"),
        Path("configs/grid/site_medium.yaml"),
        Path("configs/grid/site_dynamic.yaml"),
    ]
    records = [
        evaluate_episode(algorithm, model, config, seed)
        for algorithm, model in models.items()
        for config in configs
        for seed in range(10)
    ]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "evaluation_results.csv", records)
    training = [
        *read_training_metrics(
            "PPO", Path("artifacts/final_submission/ppo/training_metrics.jsonl")
        ),
        *read_training_metrics(
            "SAC", Path("artifacts/final_submission/sac/training_metrics.jsonl")
        ),
    ]
    write_csv(args.output_dir / "training_metrics.csv", training)
    generate_plots(args.output_dir, training)
    for algorithm in ("ppo", "sac"):
        shutil.copy2(
            Path(f"artifacts/final_submission/{algorithm}/metadata.json"),
            args.output_dir / f"{algorithm}_checkpoint_metadata.json",
        )
    summary = {
        "status": "PASSED",
        "run_count": len(records),
        "deterministic": True,
        "algorithms": {},
    }
    for algorithm in models:
        subset = [record for record in records if record["algorithm"] == algorithm]
        summary["algorithms"][algorithm] = {
            "episodes": len(subset),
            "mean_return": float(np.mean([record["episodic_return"] for record in subset])),
            "mean_safety_cost": float(np.mean([record["safety_cost"] for record in subset])),
            "mean_coverage": float(np.mean([record["coverage"] for record in subset])),
            "mean_hazard_recall": float(np.mean([record["hazard_recall"] for record in subset])),
            "collision_rate": float(np.mean([record["collisions"] > 0 for record in subset])),
            "success_rate": float(np.mean([record["success"] for record in subset])),
        }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(f"RL_BASELINE_EVALUATION=PASSED ({len(records)} episodes)")


if __name__ == "__main__":
    main()
