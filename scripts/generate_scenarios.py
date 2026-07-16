from __future__ import annotations

import argparse
import json
from pathlib import Path

from riskaware_saferrl.scenarios import generate_dataset, validate_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/scenarios"),
    )
    parser.add_argument("--seed", type=int, default=20260716)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    generate_dataset(
        args.output_dir,
        seed=args.seed,
        overwrite=args.overwrite,
    )
    results = validate_dataset(args.output_dir)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
