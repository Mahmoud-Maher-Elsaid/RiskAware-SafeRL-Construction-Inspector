from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from riskaware_saferrl.evaluation.expert_baselines import (
    evaluate_plan,
)
from riskaware_saferrl.planners import (
    build_viewpoint_inspection_plan,
)
from riskaware_saferrl.scenario_dataset import load_scenarios


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=Path("data/scenarios"),
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=[
            "train",
            "validation",
            "test_seen",
            "test_unseen",
            "stress",
        ],
    )
    parser.add_argument(
        "--inspection-radius",
        type=int,
        default=2,
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/results/viewpoint_feasibility.json"),
    )
    return parser.parse_args()


def summarize_split(
    records: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "scenario_count": len(records),
        "plan_completion_rate": float(
            np.mean([float(record["plan_complete"]) for record in records])
        ),
        "execution_success_rate": float(np.mean([float(record["success"]) for record in records])),
        "mean_hazard_recall": float(
            np.mean([float(record["hazard_recall"]) for record in records])
        ),
        "mean_safety_cost": float(np.mean([float(record["safety_cost"]) for record in records])),
        "maximum_safety_cost": float(np.max([float(record["safety_cost"]) for record in records])),
        "mean_actions": float(np.mean([float(record["planned_actions"]) for record in records])),
        "mean_viewpoints": float(
            np.mean([float(record["planned_viewpoints"]) for record in records])
        ),
    }


def main() -> None:
    args = parse_args()

    report: dict[str, Any] = {
        "planner": "safe_viewpoint_astar",
        "inspection_radius": args.inspection_radius,
        "splits": {},
    }

    for split in args.splits:
        scenarios = load_scenarios(
            args.dataset_dir,
            split,
            verify_hash=True,
        )

        records: list[dict[str, Any]] = []

        for scenario in scenarios:
            plan = build_viewpoint_inspection_plan(
                scenario,
                safety_aware=True,
                inspection_radius=args.inspection_radius,
            )
            records.append(
                evaluate_plan(
                    scenario,
                    plan,
                )
            )

        report["splits"][split] = summarize_split(records)

    args.output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    args.output.write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
    )

    print(json.dumps(report, indent=2))
    print(f"Feasibility report: {args.output}")


if __name__ == "__main__":
    main()
