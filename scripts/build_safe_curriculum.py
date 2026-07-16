from __future__ import annotations

import argparse
import json
from pathlib import Path

from riskaware_saferrl.curriculum import (
    build_curriculum_manifest,
    save_curriculum_manifest,
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
        "--train-split",
        type=str,
        default="train",
    )
    parser.add_argument(
        "--validation-split",
        type=str,
        default="validation",
    )
    parser.add_argument(
        "--inspection-radius",
        type=int,
        default=2,
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("configs/curriculum/safe_viewpoint_radius2.json"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train_scenarios = load_scenarios(
        args.dataset_dir,
        args.train_split,
        verify_hash=True,
    )
    validation_scenarios = load_scenarios(
        args.dataset_dir,
        args.validation_split,
        verify_hash=True,
    )
    manifest = build_curriculum_manifest(
        train_scenarios,
        validation_scenarios,
        source_manifest_path=args.dataset_dir / "manifest.json",
        inspection_radius=args.inspection_radius,
    )
    output_path = save_curriculum_manifest(manifest, args.output)

    print(
        json.dumps(
            {
                "output": str(output_path),
                "inspection_radius": manifest["inspection_radius"],
                "train": {
                    "total_count": manifest["splits"]["train"]["total_count"],
                    "feasible_count": manifest["splits"]["train"]["feasible_count"],
                    "excluded_count": manifest["splits"]["train"]["excluded_count"],
                    "tier_counts": manifest["splits"]["train"]["tier_counts"],
                },
                "validation": {
                    "total_count": manifest["splits"]["validation"]["total_count"],
                    "feasible_count": manifest["splits"]["validation"]["feasible_count"],
                    "excluded_count": manifest["splits"]["validation"]["excluded_count"],
                },
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
