from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def metric_record(name: str, path: Path, stage: str) -> dict[str, Any]:
    summary = read(path)
    return {
        "name": name,
        "stage": stage,
        "checkpoint": summary["checkpoint"],
        "checkpoint_sha256": summary["checkpoint_sha256"],
        "episodes": summary["episodes"],
        "overall_mission_success": summary["overall_mission_success"],
        "target_mission_success": summary["target_mission_success"],
        "hardest_condition_success": summary["hardest_condition_success"],
        "hazard_recall": summary["hazard_recall"],
        "inspection_coverage": summary["inspection_coverage"],
        "collision_rate": summary["collision_rate"],
        "invalid_actions": summary["invalid_actions"],
        "mean_safety_cost": summary["mean_safety_cost"],
        "constraint_violations": summary["constraint_violations"],
        "partial_mission_gate": summary["replacement_gate_partial"],
    }


def main() -> None:
    root = Path("reports/strong_policy_upgrade/hrmppo_v2")
    search = root / "search"
    records = []
    for index in range(6):
        name = f"trial_{index:02d}"
        records.append(
            metric_record(
                name,
                search / f"{name}_evaluation" / "summary.json",
                "unanchored_5k_search",
            )
        )
    for name in ("anchor_lr1e6", "anchor_lr5e6", "anchor_lr1e5"):
        records.append(
            metric_record(
                name,
                search / f"{name}_evaluation" / "summary.json",
                "anchored_10k",
            )
        )
        records.append(
            metric_record(
                f"{name}_medium",
                search / f"{name}_medium_evaluation" / "summary.json",
                "top_three_30k",
            )
        )
    for name in ("anchor_lr1e6", "anchor_lr5e6"):
        records.append(
            metric_record(
                f"{name}_100k",
                search / f"{name}_100k_evaluation" / "summary.json",
                "non_dominated_100k",
            )
        )
    v1 = read(Path("reports/strong_policy_upgrade/failure_diagnosis.json"))["v1_benchmark"][
        "RiskShield-PPO"
    ]
    selected = next(row for row in records if row["name"] == "anchor_lr5e6_100k")
    risk_dagger = metric_record(
        "risk_astar_dagger_iteration_1",
        root / "risk_dagger_iteration1_evaluation" / "summary.json",
        "risk_aware_dagger_repair",
    )
    all_candidates = [*records, risk_dagger]
    failure_reasons = []
    if selected["mean_safety_cost"] > v1["safety_cost"]:
        failure_reasons.append("safety_cost_worse_than_shielded_v1")
    # The v1 report does not expose an aggregate constraint count under the
    # v2 environment contract, so this comparison cannot be asserted.
    failure_reasons.append("constraint_violation_comparison_not_contract_matched")
    failure_reasons.append("five_seed_final_training_not_started_after_failed_replacement_gate")
    report = {
        "status": "FAILED",
        "replacement_authorized": False,
        "algorithm": "RiskShield-HRMPPO-v2",
        "improvement_iterations_completed": [
            "unanchored PPO search (catastrophic forgetting)",
            "KL-anchored 10k and top-three 30k retraining",
            "non-dominated 100k continuation",
            "observation-only risk-A* policy-visited DAgger",
        ],
        "v1_preserved_baseline": v1,
        "selected_preliminary_candidate": selected,
        "risk_aware_dagger_candidate": risk_dagger,
        "candidates": all_candidates,
        "failure_reasons": failure_reasons,
        "diagnosis": {
            "systematic_teacher_mean_successful_episode_safety_cost": 31.342413793103447,
            "risk_astar_teacher_mean_successful_episode_safety_cost": 8.699250288350635,
            "teacher_strategy_disagreement_rate_iteration_1": 0.5369166666666667,
            "finding": (
                "The systematic policy achieves mission gates but traverses observed risks. "
                "Risk-A* labels are substantially safer but conflict on more than half of "
                "policy-visited states; repeated correction caused mission collapse."
            ),
        },
        "downstream_gates": {
            "five_seed_final_training": "NOT_STARTED",
            "cv_audit": "NOT_STARTED",
            "webots_v2_replacement": "NOT_STARTED",
            "benchmark_v2": "NOT_STARTED",
            "paper_v2_claims": "NOT_STARTED",
        },
    }
    root.mkdir(parents=True, exist_ok=True)
    (root / "optimization_summary.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print("HRMPPO_V2_REPLACEMENT_GATE=FAILED")


if __name__ == "__main__":
    main()
