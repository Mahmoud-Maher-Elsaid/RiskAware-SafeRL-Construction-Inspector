from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from sb3_contrib import MaskablePPO
from stable_baselines3 import PPO

from riskaware_saferrl.curriculum import (
    feasible_validation_scenarios_from_manifest,
    load_curriculum_manifest,
)
from riskaware_saferrl.evaluation.scenario_evaluator import (
    evaluate_policy_on_scenarios,
    save_evaluation_results,
)
from riskaware_saferrl.scenario_dataset import load_scenarios


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--algorithm",
        choices=("ppo", "maskable_ppo"),
        default="ppo",
    )
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
        "--curriculum-manifest",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--feasible-only",
        action="store_true",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--shield",
        action="store_true",
    )
    parser.add_argument(
        "--stochastic",
        action="store_true",
    )
    parser.add_argument(
        "--output-name",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--output-directory",
        type=Path,
        default=Path("artifacts/results"),
    )

    return parser.parse_args()


def load_model(
    model_path: Path,
    algorithm: str,
) -> PPO | MaskablePPO:
    if algorithm == "maskable_ppo":
        return MaskablePPO.load(model_path)

    return PPO.load(model_path)


def select_scenarios(
    args: argparse.Namespace,
) -> tuple[Any, ...]:
    scenarios = tuple(
        load_scenarios(
            args.dataset_dir,
            args.split,
            verify_hash=True,
        )
    )

    if args.feasible_only:
        if args.split != "validation":
            raise ValueError("--feasible-only currently supports validation only.")

        if args.curriculum_manifest is None:
            raise ValueError("--curriculum-manifest is required with --feasible-only.")

        manifest = load_curriculum_manifest(
            args.curriculum_manifest,
        )
        scenarios = feasible_validation_scenarios_from_manifest(
            scenarios,
            manifest,
        )

    if args.limit is not None:
        if args.limit < 1:
            raise ValueError("--limit must be positive.")

        scenarios = scenarios[: args.limit]

    if not scenarios:
        raise ValueError("No scenarios were selected.")

    return scenarios


def main() -> None:
    args = parse_args()

    model = load_model(
        args.model,
        args.algorithm,
    )
    scenarios = select_scenarios(args)

    use_action_masks = args.algorithm == "maskable_ppo"

    records, summary = evaluate_policy_on_scenarios(
        model,
        scenarios,
        use_shield=args.shield,
        use_action_masks=use_action_masks,
        deterministic=not args.stochastic,
    )

    summary["algorithm"] = args.algorithm
    summary["model"] = str(args.model)
    summary["feasible_only"] = args.feasible_only

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
