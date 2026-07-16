from __future__ import annotations

import argparse
import json
from pathlib import Path

from riskaware_saferrl.planners import (
    build_oracle_inspection_plan,
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
        default="train",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/expert/train_oracle_plans.jsonl"),
    )
    return parser.parse_args()


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

    args.output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    complete_count = 0

    with args.output.open(
        "w",
        encoding="utf-8",
        newline="\n",
    ) as file:
        for scenario in scenarios:
            plan = build_oracle_inspection_plan(
                scenario,
                safety_aware=True,
            )

            complete_count += int(plan.complete)

            record = {
                "scenario_id": scenario.scenario_id,
                "split": scenario.split,
                "planner": plan.planner_name,
                "actions": list(plan.actions),
                "visited_hazards": [list(position) for position in plan.visited_hazards],
                "unreachable_hazards": [list(position) for position in plan.unreachable_hazards],
                "complete": plan.complete,
            }

            file.write(
                json.dumps(
                    record,
                    separators=(",", ":"),
                )
                + "\n"
            )

    print(
        json.dumps(
            {
                "output": str(args.output),
                "scenario_count": len(scenarios),
                "complete_plans": complete_count,
                "completion_rate": (complete_count / len(scenarios)),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
