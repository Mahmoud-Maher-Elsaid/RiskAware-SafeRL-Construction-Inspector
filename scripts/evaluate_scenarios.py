from __future__ import annotations

import argparse
import json
from pathlib import Path

from stable_baselines3 import PPO

from riskaware_saferrl.evaluation.scenario_evaluator import (
    evaluate_policy_on_scenarios,
    save_evaluation_results,
)
from riskaware_saferrl.scenario_dataset import load_scenarios


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=Path("data/scenarios"),
    )
    parser.add_argument("--split", type=str, default="validation")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--shield", action="store_true")
    parser.add_argument("--output-name", type=str, default=None)
    parser.add_argument(
        "--output-directory",
        type=Path,
        default=Path("artifacts/results"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model = PPO.load(args.model)

    scenarios = load_scenarios(
        args.dataset_dir,
        args.split,
        verify_hash=True,
    )

    if args.limit is not None:
        if args.limit < 1:
            raise ValueError("--limit must be positive.")

        scenarios = scenarios[: args.limit]

    records, summary = evaluate_policy_on_scenarios(
        model,
        scenarios,
        use_shield=args.shield,
        deterministic=True,
    )

    output_name = args.output_name or f"{args.model.stem}_{args.split}"
    csv_path, json_path = save_evaluation_results(
        records,
        summary,
        args.output_directory,
        output_name,
    )

    print(json.dumps(summary, indent=2))
    print(f"Episode results: {csv_path}")
    print(f"Summary: {json_path}")


if __name__ == "__main__":
    main()
