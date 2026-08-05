"""Create a truthful v4 ablation record from executable evidence."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    source = root / "reports/strong_policy_upgrade/benchmark_v2/hierarchical_v4_rows.csv"
    out = root / "reports/strong_policy_upgrade/ablations_v4"
    out.mkdir(parents=True, exist_ok=True)
    rows = pd.read_csv(source)
    rows.insert(1, "variant", "full_hierarchical_experimental")
    rows.to_csv(out / "raw_results.csv", index=False)
    rows.to_parquet(out / "raw_results.parquet", index=False)
    summary = (
        rows.groupby("variant")
        .agg(
            mission_success=("success", "mean"),
            hazard_recall=("hazard_recall", "mean"),
            coverage=("inspection_coverage", "mean"),
            safety_cost=("safety_cost", "mean"),
            shield_intervention=("shield_interventions", "mean"),
            planner_failures=("planner_failures", "sum"),
        )
        .reset_index()
    )
    summary.to_csv(out / "summary.csv", index=False)
    infeasible = {}
    for variant in [
        "heuristic_option_selector",
        "no_recurrent_memory",
        "no_target_persistence",
        "no_causal_imitation",
        "no_constrained_option_ppo",
        "no_target_head",
        "fixed_duration",
        "static_planner",
        "no_worker_prediction",
        "no_predictive_controller",
        "no_safety_shield",
        "one_step_shield",
        "no_cv_risk",
        "no_kl_anchoring",
    ]:
        infeasible[variant] = {
            "status": "technically_infeasible",
            "reason": "No validated alternate checkpoint or inference-time toggle exists; generating metrics would fabricate an experiment.",
        }
    payload = {
        "executed_variants": ["full_hierarchical_experimental"],
        "infeasible_variants": infeasible,
        "paired_seed_count": 270,
    }
    (out / "summary.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    (out / "statistics.json").write_text(
        json.dumps(
            {
                "method": "descriptive metrics; no unsupported significance claim",
                "sample_count": 270,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    (out / "statistics.md").write_text(
        "# v4 Ablation Statistics\n\nOnly the full executable hierarchical system was run. Other variants are explicitly infeasible because no validated toggle or checkpoint exists.\n",
        encoding="utf-8",
    )
    (out / "infeasible_variants.json").write_text(
        json.dumps(infeasible, indent=2) + "\n", encoding="utf-8"
    )
    (out / "protocol.md").write_text(
        "# v4 Ablation Protocol\n\nThe full system uses the verified H4 checkpoint, causal planner, predictive controller, and Safety Shield. Unsupported variants are not assigned fabricated metrics.\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
