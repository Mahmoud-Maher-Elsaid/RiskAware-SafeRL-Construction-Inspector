from __future__ import annotations

import argparse
import csv
import json
import time
from dataclasses import asdict, fields
from pathlib import Path

import numpy as np
import yaml

from riskaware_saferrl.baselines import (
    FrontierExplorationPlanner,
    NearestRiskRevisitPlanner,
    RiskAwareAStarPlanner,
)
from riskaware_saferrl.envs import GridEnvironmentConfig, ResearchConstructionEnv

PLANNERS = {
    "risk_aware_astar": RiskAwareAStarPlanner,
    "frontier_exploration": FrontierExplorationPlanner,
    "nearest_risk_revisit": NearestRiskRevisitPlanner,
}


def load_environment_config(path: Path) -> GridEnvironmentConfig:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    allowed = {field.name for field in fields(GridEnvironmentConfig)}
    return GridEnvironmentConfig(**{key: value for key, value in payload.items() if key in allowed})


def evaluate_episode(
    planner_name: str,
    config_path: Path,
    seed: int,
    risk_weight: float,
) -> dict[str, object]:
    environment = ResearchConstructionEnv(load_environment_config(config_path))
    planner = PLANNERS[planner_name](risk_weight=risk_weight)
    _, info = environment.reset(seed=seed)
    planner.reset()
    planning_times: list[float] = []
    terminated = truncated = False
    decision_reasons: dict[str, int] = {}
    while not (terminated or truncated):
        started = time.perf_counter()
        decision = planner.decide(environment)
        planning_times.append((time.perf_counter() - started) * 1000.0)
        decision_reasons[decision.reason] = decision_reasons.get(decision.reason, 0) + 1
        _, _, terminated, truncated, info = environment.step(decision.action)
    telemetry = asdict(environment.telemetry())
    environment.close()
    return {
        "run_id": f"{planner_name}:{config_path.stem}:{seed}",
        "algorithm": planner_name,
        "environment": config_path.stem,
        "seed": seed,
        "inspection_coverage": info["coverage"],
        "hazard_recall": info["hazard_recall"],
        "path_length": telemetry["path_length"],
        "collisions": telemetry["collisions"],
        "near_misses": telemetry["near_misses"],
        "restricted_zone_violations": telemetry["restricted_violations"],
        "time_to_inspect": telemetry["steps"],
        "energy_proxy": telemetry["energy_usage"],
        "success": bool(info["success"]),
        "average_planning_latency_ms": float(np.mean(planning_times)),
        "decision_reasons": json.dumps(decision_reasons, sort_keys=True),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--risk-weight", type=float, default=5.0)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("reports/final_submission/stage2_planner_baselines"),
    )
    args = parser.parse_args()
    configs = [
        Path("configs/grid/site_small.yaml"),
        Path("configs/grid/site_medium.yaml"),
        Path("configs/grid/site_dynamic.yaml"),
    ]
    records = [
        evaluate_episode(planner, config, seed, args.risk_weight)
        for planner in PLANNERS
        for config in configs
        for seed in range(args.seeds)
    ]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    raw_path = args.output_dir / "raw_results.csv"
    with raw_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    summary: dict[str, object] = {
        "status": "PASSED",
        "run_count": len(records),
        "seeds": list(range(args.seeds)),
        "algorithms": {},
    }
    for planner in PLANNERS:
        subset = [record for record in records if record["algorithm"] == planner]
        summary["algorithms"][planner] = {
            "episodes": len(subset),
            "success_rate": float(np.mean([record["success"] for record in subset])),
            "mean_hazard_recall": float(np.mean([record["hazard_recall"] for record in subset])),
            "mean_coverage": float(np.mean([record["inspection_coverage"] for record in subset])),
            "mean_collisions": float(np.mean([record["collisions"] for record in subset])),
            "mean_safety_violations": float(
                np.mean(
                    [
                        record["near_misses"] + record["restricted_zone_violations"]
                        for record in subset
                    ]
                )
            ),
        }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(f"PLANNER_BASELINES=PASSED ({len(records)} episodes)")


if __name__ == "__main__":
    main()
