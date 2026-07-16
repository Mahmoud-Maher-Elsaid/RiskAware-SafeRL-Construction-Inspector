from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from riskaware_saferrl.evaluation.expert_baselines import (
    evaluate_all_baselines,
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
        "--split",
        type=str,
        default="validation",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=42,
    )
    parser.add_argument(
        "--inspection-radius",
        type=int,
        default=2,
    )
    parser.add_argument(
        "--output-directory",
        type=Path,
        default=Path("artifacts/results/expert_baselines"),
    )
    return parser.parse_args()


def summarize_records(
    records: list[dict[str, Any]],
) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)

    for record in records:
        grouped[str(record["planner"])].append(record)

    metrics = (
        "reward",
        "safety_cost",
        "hazard_recall",
        "coverage",
        "success",
        "steps",
        "plan_complete",
        "step_penalty",
        "new_cell_reward",
        "hazard_reward",
        "completion_reward",
        "collision_penalty",
        "worker_penalty",
        "restricted_penalty",
        "invalid_inspection_penalty",
    )

    summary: dict[str, Any] = {}

    for planner_name, planner_records in sorted(grouped.items()):
        summary[planner_name] = {
            "scenario_count": len(planner_records),
            "metrics": {
                metric_name: {
                    "mean": float(
                        np.mean([float(record.get(metric_name, 0.0)) for record in planner_records])
                    ),
                    "std": float(
                        np.std(
                            [float(record.get(metric_name, 0.0)) for record in planner_records],
                            ddof=1,
                        )
                    )
                    if len(planner_records) > 1
                    else 0.0,
                }
                for metric_name in metrics
            },
        }

    return summary


def main() -> None:
    args = parse_args()

    scenarios = load_scenarios(
        args.dataset_dir,
        args.split,
        verify_hash=True,
    )

    if args.limit is not None:
        if args.limit < 1:
            raise ValueError("--limit must be positive.")

        scenarios = scenarios[: args.limit]

    records = evaluate_all_baselines(
        scenarios,
        random_seed=args.random_seed,
        inspection_radius=args.inspection_radius,
    )
    summary = summarize_records(records)

    args.output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    csv_path = args.output_directory / f"{args.split}_baseline_episodes.csv"
    json_path = args.output_directory / f"{args.split}_baseline_summary.json"

    with csv_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=sorted({key for record in records for key in record}),
        )
        writer.writeheader()
        writer.writerows(records)

    json_path.write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )

    print(json.dumps(summary, indent=2))
    print(f"Episode results: {csv_path}")
    print(f"Summary: {json_path}")


if __name__ == "__main__":
    main()
