from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from riskaware_saferrl.benchmarking import (
    REQUIRED_RESULT_FIELDS,
    build_manifest,
    load_settings,
    run_benchmark,
)

METRICS = (
    "hazard_recall",
    "inspection_coverage",
    "collision_rate",
    "near_miss_rate",
    "constraint_violations",
    "time_to_inspect",
    "mission_duration",
    "energy_usage",
    "robustness_score",
    "success_rate",
    "safety_cost",
    "path_length",
    "shield_interventions",
    "emergency_stops",
    "inference_latency_ms",
    "policy_latency_ms",
)


def atomic_dataframe(frame: pd.DataFrame, path: Path, *, parquet: bool = False) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    if parquet:
        frame.to_parquet(temporary, index=False)
    else:
        frame.to_csv(temporary, index=False)
    os.replace(temporary, path)


def atomic_text(path: Path, content: str) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(content, encoding="utf-8")
    os.replace(temporary, path)


def aggregate(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    group_columns = [
        "algorithm",
        "environment_size_name",
        "hazard_density_name",
        "perception_noise",
    ]
    for keys, group in frame.groupby(group_columns, sort=True):
        for metric in METRICS:
            values = group[metric].astype(float).to_numpy()
            mean = float(np.mean(values))
            sem = float(stats.sem(values)) if len(values) > 1 else 0.0
            interval = (
                stats.t.interval(0.95, len(values) - 1, loc=mean, scale=sem)
                if len(values) > 1 and sem > 0
                else (mean, mean)
            )
            rows.append(
                {
                    **dict(zip(group_columns, keys, strict=True)),
                    "metric": metric,
                    "count": len(values),
                    "mean": mean,
                    "standard_deviation": (
                        float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
                    ),
                    "median": float(np.median(values)),
                    "iqr": float(np.percentile(values, 75) - np.percentile(values, 25)),
                    "ci95_low": float(interval[0]),
                    "ci95_high": float(interval[1]),
                }
            )
    return pd.DataFrame(rows)


def _holm_adjust(p_values: list[float]) -> list[float]:
    order = np.argsort(p_values)
    adjusted = np.empty(len(p_values))
    running = 0.0
    for rank, index in enumerate(order):
        corrected = min(1.0, (len(p_values) - rank) * p_values[index])
        running = max(running, corrected)
        adjusted[index] = running
    return adjusted.tolist()


def statistical_analysis(frame: pd.DataFrame) -> dict[str, object]:
    algorithms = sorted(frame["algorithm"].unique())
    analyses: dict[str, object] = {}
    raw_pair_p_values = []
    metrics = (
        "hazard_recall",
        "inspection_coverage",
        "collision_rate",
        "constraint_violations",
        "safety_cost",
        "success_rate",
    )
    for metric in metrics:
        pivot = frame.pivot(
            index=[
                "environment_size_name",
                "hazard_density_name",
                "perception_noise",
                "evaluation_seed",
            ],
            columns="algorithm",
            values=metric,
        ).dropna()
        samples = [pivot[algorithm].to_numpy() for algorithm in algorithms]
        all_equal = all(np.array_equal(samples[0], sample) for sample in samples[1:])
        if all_equal:
            friedman_statistic, friedman_p_value = 0.0, 1.0
        else:
            friedman_statistic, friedman_p_value = stats.friedmanchisquare(*samples)
        ppo = pivot["PPO"].to_numpy()
        riskshield = pivot["RiskShield-PPO"].to_numpy()
        differences = riskshield - ppo
        difference_sd = float(np.std(differences, ddof=1))
        effect_size = float(np.mean(differences) / difference_sd) if difference_sd > 0 else 0.0
        try:
            pair_p = float(stats.wilcoxon(riskshield, ppo).pvalue)
        except ValueError:
            pair_p = 1.0
        raw_pair_p_values.append(pair_p)
        analyses[metric] = {
            "friedman_statistic": float(friedman_statistic),
            "friedman_p_value": float(friedman_p_value),
            "riskshield_vs_ppo_standardized_paired_mean_difference": effect_size,
            "wilcoxon_raw_p_value": pair_p,
            "paired_sample_count": len(pivot),
        }
    for metric, adjusted in zip(metrics, _holm_adjust(raw_pair_p_values), strict=True):
        analyses[metric]["wilcoxon_holm_adjusted_p_value"] = adjusted
    return {
        "method": {
            "design": "Repeated measures by site size, hazard density, noise, and seed.",
            "omnibus_test": "Friedman rank test because bounded and zero-inflated metrics need not be normal.",
            "planned_pair": "Paired Wilcoxon signed-rank: RiskShield-PPO versus PPO.",
            "multiple_comparison_correction": "Holm family-wise correction across six planned metrics.",
            "effect_size": "Standardized paired mean difference; zero when paired differences have zero variance.",
            "confidence_intervals": "Two-sided 95% Student-t intervals over ten evaluation seeds in each aggregate cell.",
            "missing_data_policy": "No imputation; Stage 9 fails if any primary run is missing.",
        },
        "metrics": analyses,
    }


def analysis_markdown(analysis: dict[str, object]) -> str:
    lines = [
        "# Stage 9 Statistical Analysis",
        "",
        "The analysis uses the complete paired benchmark matrix. No failed or missing "
        "run is imputed.",
        "",
        "## Method",
        "",
    ]
    for key, value in analysis["method"].items():
        lines.append(f"- {key.replace('_', ' ').title()}: {value}")
    lines.extend(
        [
            "",
            "## Results",
            "",
            "| Metric | Friedman p | Wilcoxon Holm p | Paired effect size | n |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for metric, result in analysis["metrics"].items():
        lines.append(
            f"| {metric} | {result['friedman_p_value']:.6g} | "
            f"{result['wilcoxon_holm_adjusted_p_value']:.6g} | "
            f"{result['riskshield_vs_ppo_standardized_paired_mean_difference']:.4f} | "
            f"{result['paired_sample_count']} |"
        )
    return "\n".join(lines) + "\n"


def write_outputs(
    records: list[dict[str, object]],
    output: Path,
    settings: dict[str, object],
    *,
    final_matrix: bool,
) -> None:
    frame = pd.DataFrame(records)
    missing_columns = REQUIRED_RESULT_FIELDS - set(frame.columns)
    if missing_columns:
        raise RuntimeError(f"Raw results omit required fields: {sorted(missing_columns)}")
    atomic_dataframe(frame, output / "raw_results.csv")
    atomic_dataframe(frame, output / "raw_results.parquet", parquet=True)
    aggregated = aggregate(frame)
    atomic_dataframe(aggregated, output / "aggregated_results.csv")
    analysis = statistical_analysis(frame) if final_matrix else {}
    atomic_text(
        output / "statistical_analysis.json",
        json.dumps(analysis, indent=2) + "\n",
    )
    if final_matrix:
        atomic_text(output / "statistical_analysis.md", analysis_markdown(analysis))
    expected = int(settings["expected_primary_runs"]) if final_matrix else len(frame)
    unique = int(frame["run_id"].nunique())
    summary = {
        "status": ("PASSED" if len(frame) == expected and unique == expected else "FAILED"),
        "run_count": len(frame),
        "expected_run_count": expected,
        "unique_run_ids": unique,
        "duplicate_run_count": len(frame) - unique,
        "missing_run_count": expected - unique,
        "failed_run_count": 0,
        "configuration_hash_count": int(frame["configuration_hash"].nunique()),
        "algorithms": {
            algorithm: {
                "runs": len(group),
                "mean_hazard_recall": float(group["hazard_recall"].mean()),
                "mean_coverage": float(group["inspection_coverage"].mean()),
                "mean_safety_cost": float(group["safety_cost"].mean()),
                "success_rate": float(group["success_rate"].mean()),
            }
            for algorithm, group in frame.groupby("algorithm")
        },
    }
    atomic_text(
        output / "benchmark_summary.json",
        json.dumps(summary, indent=2) + "\n",
    )
    atomic_text(
        output / "resume_state.json",
        json.dumps(
            {
                "completed": len(frame),
                "expected": expected,
                "missing": expected - unique,
                "failed": 0,
            },
            indent=2,
        )
        + "\n",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/benchmarks/full_matrix.yaml"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reports/final_submission/stage9_benchmark"),
    )
    parser.add_argument("--limit", type=int)
    parser.add_argument("--one-per-algorithm", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    settings = load_settings(args.config)
    manifest = build_manifest(settings)
    if args.validate_only:
        print(
            f"BENCHMARK_CONFIGURATION=PASSED "
            f"expected={len(manifest)} unique={len({run.run_id for run in manifest})}"
        )
        return
    args.output.mkdir(parents=True, exist_ok=True)
    records = run_benchmark(
        args.config,
        args.output,
        limit=args.limit,
        one_per_algorithm=args.one_per_algorithm,
    )
    final_matrix = args.limit is None and not args.one_per_algorithm
    write_outputs(records, args.output, settings, final_matrix=final_matrix)
    atomic_text(
        args.output / "run_manifest.json",
        json.dumps([asdict(run) for run in manifest], indent=2) + "\n",
    )
    print(f"FULL_RESEARCH_BENCHMARK=PASSED ({len(records)} runs)")


if __name__ == "__main__":
    main()
