from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np

from riskaware_saferrl.envs import (
    ConstructionInspectionEnv,
)
from riskaware_saferrl.safety import SafetyShield
from riskaware_saferrl.scenario_dataset import (
    load_scenarios,
)


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
        default=("train", "validation"),
    )
    parser.add_argument(
        "--steps-per-scenario",
        type=int,
        default=100,
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.steps_per_scenario < 1:
        raise ValueError("--steps-per-scenario must be positive.")

    rng = np.random.default_rng(args.seed)
    resolution_counts: Counter[str] = Counter()
    total_steps = 0
    total_cost = 0.0
    scenario_count = 0

    for split in args.splits:
        scenarios = load_scenarios(
            args.dataset_dir,
            split,
            verify_hash=True,
        )

        for scenario in scenarios:
            scenario_count += 1
            environment = SafetyShield(
                ConstructionInspectionEnv(
                    scenario=scenario,
                    inspection_radius=2,
                )
            )
            environment.reset(seed=args.seed)

            for _ in range(args.steps_per_scenario):
                valid_actions = np.flatnonzero(environment.action_masks())
                proposed_action = int(rng.choice(valid_actions))

                (
                    _,
                    _,
                    terminated,
                    truncated,
                    info,
                ) = environment.step(proposed_action)

                total_steps += 1
                total_cost += float(info.get("cost", 0.0))
                resolution_counts[
                    str(
                        info.get(
                            "shield_resolution",
                            "missing",
                        )
                    )
                ] += 1

                if terminated or truncated:
                    environment.reset(seed=args.seed)

            environment.close()

    summary = {
        "scenario_count": scenario_count,
        "total_steps": total_steps,
        "total_cost": total_cost,
        "resolution_counts": dict(sorted(resolution_counts.items())),
    }

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
