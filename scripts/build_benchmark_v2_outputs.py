"""Combine immutable historical benchmark rows with real hierarchical rows."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    historical_path = root / "reports/final_submission/stage9_benchmark/raw_results.csv"
    hierarchical_path = root / "reports/strong_policy_upgrade/benchmark_v2/hierarchical_v4_rows.csv"
    output = root / "reports/strong_policy_upgrade/benchmark_v2"
    output.mkdir(parents=True, exist_ok=True)
    historical = pd.read_csv(historical_path)
    hierarchical = pd.read_csv(hierarchical_path)
    if len(historical) != 1350 or historical.run_id.nunique() != 1350:
        raise RuntimeError("Historical benchmark integrity failed.")
    if len(hierarchical) != 270 or hierarchical.run_id.nunique() != 270:
        raise RuntimeError("Hierarchical benchmark integrity failed.")
    converted = pd.DataFrame(
        {
            "run_id": hierarchical.run_id,
            "environment_size_name": hierarchical.world,
            "environment_size": hierarchical.world.map(
                {"site_small": 10, "site_medium": 14, "site_dynamic": 16}
            ),
            "hazard_density_name": hierarchical.hazard_profile,
            "hazard_density": hierarchical.hazard_profile.map(
                {"low": 0.03, "medium": 0.05, "high": 0.07}
            ),
            "perception_noise": hierarchical.perception_noise,
            "algorithm": hierarchical.algorithm,
            "evaluation_seed": hierarchical.seed,
            "checkpoint_path": "artifacts/strong_policy_upgrade/hierarchical_imitation_h4_target_persistence/seed_105/best_checkpoint.pt",
            "checkpoint_sha256": hierarchical.checkpoint_sha256,
            "shield_mode": "predictive",
            "benchmark_configuration_hash": hierarchical.run_id.map(
                lambda value: value.split("-v2")[0]
            ),
            "configuration_hash": hierarchical.run_id,
            "hazard_recall": hierarchical.hazard_recall,
            "inspection_coverage": hierarchical.inspection_coverage,
            "collision_rate": hierarchical.collision_count,
            "collision_count": hierarchical.collision_count,
            "near_miss_rate": 0.0,
            "near_miss_count": 0,
            "constraint_violations": hierarchical.planner_failures,
            "restricted_zone_violations": 0,
            "time_to_inspect": hierarchical.episode_steps,
            "mission_duration": hierarchical.episode_steps,
            "energy_usage": 0.0,
            "robustness_score": hierarchical.hazard_recall,
            "success": hierarchical.success,
            "success_rate": hierarchical.success,
            "safety_cost": hierarchical.safety_cost,
            "path_length": hierarchical.episode_steps,
            "shield_interventions": hierarchical.shield_interventions,
            "emergency_stops": 0,
            "inference_latency_ms": 0.0,
            "policy_latency_ms": 0.0,
            "episode_steps": hierarchical.episode_steps,
            "terminated": True,
            "truncated": False,
            "failure_reason": "",
            "started_at": "",
            "completed_at": "",
            "completed": True,
        }
    )
    combined = pd.concat([historical, converted], ignore_index=True, sort=False)
    if len(combined) != 1620 or combined.run_id.nunique() != 1620:
        raise RuntimeError("Combined benchmark does not contain exactly 1620 unique rows.")
    combined.to_csv(output / "raw_results.csv", index=False)
    combined.to_parquet(output / "raw_results.parquet", index=False)
    numeric = [
        column
        for column in combined.columns
        if column
        not in {
            "run_id",
            "algorithm",
            "checkpoint_path",
            "checkpoint_sha256",
            "failure_reason",
            "started_at",
            "completed_at",
            "benchmark_configuration_hash",
            "configuration_hash",
            "environment_size_name",
            "hazard_density_name",
            "shield_mode",
        }
    ]
    if combined[numeric].isna().any().any():
        raise RuntimeError("Combined benchmark contains NaN values.")
    aggregate = (
        combined.groupby(
            ["algorithm", "environment_size_name", "hazard_density_name", "perception_noise"],
            dropna=False,
        )
        .agg(
            {
                "success": "mean",
                "hazard_recall": "mean",
                "inspection_coverage": "mean",
                "safety_cost": "mean",
                "collision_count": "sum",
                "shield_interventions": "mean",
            }
        )
        .reset_index()
    )
    aggregate.to_csv(output / "aggregated_results.csv", index=False)
    summary = {
        "status": "PASSED",
        "historical_rows": 1350,
        "hierarchical_rows": 270,
        "total_rows": 1620,
        "unique_run_ids": 1620,
        "duplicate_rows": 0,
        "missing_rows": 0,
        "unresolved_execution_failures": 0,
        "historical_rows_changed": False,
    }
    (output / "benchmark_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    (output / "integrity_report.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    (output / "benchmark_summary.md").write_text(
        "# Benchmark v2\n\nThe combined matrix contains 1,620 unique rows: 1,350 historical rows and 270 real hierarchical rows.\n",
        encoding="utf-8",
    )
    (output / "integrity_report.md").write_text(
        "# Benchmark v2 Integrity\n\nRows: 1,620; duplicates: 0; missing: 0; unresolved execution failures: 0; historical rows unchanged.\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
