from __future__ import annotations

import argparse
import json
import subprocess
import sys
from itertools import product
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--initial-checkpoint", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=50_000)
    parser.add_argument("--seed", type=int, default=20260730)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/strong_policy_upgrade/hrmppo_v2_search"),
    )
    parser.add_argument(
        "--report-dir",
        type=Path,
        default=Path("reports/strong_policy_upgrade/hrmppo_v2/search"),
    )
    args = parser.parse_args()
    trials = []
    for index, (learning_rate, safety_budget) in enumerate(
        product((5e-5, 1e-4, 2e-4), (15.0, 20.0))
    ):
        trial_name = f"trial_{index:02d}"
        command = [
            sys.executable,
            "scripts/train_riskshield_hrmppo_v2.py",
            "--initial-checkpoint",
            str(args.initial_checkpoint),
            "--total-steps",
            str(args.steps),
            "--learning-rate",
            str(learning_rate),
            "--safety-budget",
            str(safety_budget),
            "--seed",
            str(args.seed + index),
            "--output-dir",
            str(args.output_dir / trial_name),
            "--report-dir",
            str(args.report_dir / trial_name),
        ]
        completed = subprocess.run(command, check=False)
        trials.append(
            {
                "trial": trial_name,
                "learning_rate": learning_rate,
                "safety_budget": safety_budget,
                "seed": args.seed + index,
                "return_code": completed.returncode,
            }
        )
    args.report_dir.mkdir(parents=True, exist_ok=True)
    (args.report_dir / "search_manifest.json").write_text(
        json.dumps({"trials": trials}, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
