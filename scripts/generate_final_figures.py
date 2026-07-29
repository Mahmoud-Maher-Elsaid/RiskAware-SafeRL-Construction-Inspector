from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path("reports/final_submission/stage9_benchmark")
FIGURES = ROOT / "figures"
TABLES = ROOT / "tables"
SOURCE = FIGURES / "source_data"


def save_bar(
    frame: pd.DataFrame,
    metric: str,
    title: str,
    ylabel: str,
    filename: str,
) -> None:
    source = (
        frame.groupby("algorithm", as_index=False)[metric]
        .agg(["mean", "std"])
        .reset_index()
        .sort_values("mean", ascending=False)
    )
    source.to_csv(SOURCE / f"{filename}.csv", index=False)
    figure, axis = plt.subplots(figsize=(8, 4.8))
    axis.bar(
        source["algorithm"],
        source["mean"],
        yerr=source["std"].fillna(0),
        capsize=3,
        color="#2f6f8f",
    )
    axis.set_title(title)
    axis.set_ylabel(ylabel)
    axis.tick_params(axis="x", rotation=25)
    figure.tight_layout()
    figure.savefig(FIGURES / f"{filename}.png", dpi=180)
    plt.close(figure)


def save_training_curve(frame: pd.DataFrame, metric: str, title: str, filename: str) -> None:
    source = frame[["algorithm", "timesteps", metric]].copy()
    source.to_csv(SOURCE / f"{filename}.csv", index=False)
    figure, axis = plt.subplots(figsize=(8, 4.8))
    for algorithm, group in source.groupby("algorithm"):
        ordered = group.sort_values("timesteps")
        smoothed = ordered[metric].rolling(10, min_periods=1).mean()
        axis.plot(ordered["timesteps"], smoothed, label=algorithm)
    axis.set_title(title)
    axis.set_xlabel("Training timesteps")
    axis.set_ylabel(metric.replace("_", " ").title())
    axis.legend()
    figure.tight_layout()
    figure.savefig(FIGURES / f"{filename}.png", dpi=180)
    plt.close(figure)


def main() -> None:
    for directory in (FIGURES, TABLES, SOURCE):
        directory.mkdir(parents=True, exist_ok=True)
    raw = pd.read_csv(ROOT / "raw_results.csv")
    stage3 = pd.read_csv("reports/final_submission/stage3_rl_baselines/training_metrics.csv")
    stage4 = pd.read_csv(
        "reports/final_submission/stage4_riskshield_ppo/constraint_training_metrics.csv"
    )
    stage4["algorithm"] = "RiskShield-PPO"
    training = pd.concat([stage3, stage4], ignore_index=True, sort=False)
    save_training_curve(
        training, "episodic_return", "Training Return (10-Episode Mean)", "learning_curves"
    )
    save_training_curve(
        training, "safety_cost", "Training Safety Cost (10-Episode Mean)", "safety_cost_curves"
    )
    plots = (
        ("hazard_recall", "Hazard Recall by Algorithm", "Recall", "hazard_recall_comparison"),
        (
            "inspection_coverage",
            "Inspection Coverage by Algorithm",
            "Coverage",
            "inspection_coverage_comparison",
        ),
        ("collision_rate", "Collision Rate by Algorithm", "Rate", "collision_comparison"),
        (
            "constraint_violations",
            "Constraint Violations by Algorithm",
            "Count",
            "constraint_violation_comparison",
        ),
        (
            "time_to_inspect",
            "Time to Inspect by Algorithm",
            "Steps",
            "time_to_inspect_comparison",
        ),
        ("energy_usage", "Energy Proxy by Algorithm", "Energy", "energy_comparison"),
        (
            "robustness_score",
            "Robustness by Algorithm",
            "Score",
            "robustness_comparison",
        ),
        (
            "shield_interventions",
            "Shield Interventions by Algorithm",
            "Count",
            "safety_shield_interventions",
        ),
        ("success_rate", "Mission Success by Algorithm", "Rate", "success_rate_comparison"),
    )
    for arguments in plots:
        save_bar(raw, *arguments)

    pareto = (
        raw.groupby("algorithm", as_index=False)
        .agg(
            inspection_coverage=("inspection_coverage", "mean"), safety_cost=("safety_cost", "mean")
        )
        .sort_values("algorithm")
    )
    pareto.to_csv(SOURCE / "safety_performance_pareto.csv", index=False)
    figure, axis = plt.subplots(figsize=(7, 5))
    axis.scatter(pareto["safety_cost"], pareto["inspection_coverage"], s=70)
    for row in pareto.itertuples():
        axis.annotate(row.algorithm, (row.safety_cost, row.inspection_coverage))
    axis.set_xlabel("Mean safety cost (lower is safer)")
    axis.set_ylabel("Mean inspection coverage (higher is better)")
    axis.set_title("Safety–Performance Pareto View")
    figure.tight_layout()
    figure.savefig(FIGURES / "safety_performance_pareto.png", dpi=180)
    plt.close(figure)

    uncertainty = (
        raw.groupby(["algorithm", "perception_noise"], as_index=False)
        .agg(hazard_recall=("hazard_recall", "mean"), robustness_score=("robustness_score", "mean"))
        .sort_values(["algorithm", "perception_noise"])
    )
    uncertainty.to_csv(SOURCE / "uncertainty_degradation.csv", index=False)
    figure, axis = plt.subplots(figsize=(8, 4.8))
    for algorithm, group in uncertainty.groupby("algorithm"):
        axis.plot(group["perception_noise"], group["hazard_recall"], marker="o", label=algorithm)
    axis.set_xlabel("Perception false-negative probability")
    axis.set_ylabel("Mean hazard recall")
    axis.set_title("Perception-Uncertainty Degradation")
    axis.legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(FIGURES / "uncertainty_degradation.png", dpi=180)
    plt.close(figure)

    algorithm_table = (
        raw.groupby("algorithm", as_index=False)
        .agg(
            runs=("run_id", "count"),
            hazard_recall=("hazard_recall", "mean"),
            inspection_coverage=("inspection_coverage", "mean"),
            collision_rate=("collision_rate", "mean"),
            safety_cost=("safety_cost", "mean"),
            success_rate=("success_rate", "mean"),
            robustness_score=("robustness_score", "mean"),
        )
        .sort_values("algorithm")
    )
    algorithm_table.to_csv(TABLES / "algorithm_summary.csv", index=False)
    (TABLES / "algorithm_summary.tex").write_text(
        algorithm_table.to_latex(index=False, float_format="%.4f"),
        encoding="utf-8",
    )
    ablation = pd.read_csv(ROOT / "ablation_results.csv")
    ablation_table = (
        ablation.groupby("variant", as_index=False)
        .agg(
            episodes=("run_id", "count"),
            hazard_recall=("hazard_recall", "mean"),
            inspection_coverage=("inspection_coverage", "mean"),
            safety_cost=("safety_cost", "mean"),
            collision_rate=("collision_count", lambda values: (values > 0).mean()),
            success_rate=("success", "mean"),
        )
        .sort_values("variant")
    )
    ablation_table.to_csv(TABLES / "ablation_summary.csv", index=False)
    (TABLES / "ablation_summary.tex").write_text(
        ablation_table.to_latex(index=False, float_format="%.4f"),
        encoding="utf-8",
    )
    print(f"FINAL_FIGURES=PASSED ({len(list(FIGURES.glob('*.png')))} figures)")


if __name__ == "__main__":
    main()
