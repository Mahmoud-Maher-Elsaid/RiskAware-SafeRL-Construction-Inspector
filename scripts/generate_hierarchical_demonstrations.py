from __future__ import annotations

import argparse
import json
from pathlib import Path

from riskaware_saferrl.training.hierarchical_demonstrations import (
    HierarchicalDatasetBuilder,
    HierarchicalDerivationConfig,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Derive observation-consistent hierarchical option demonstrations."
    )
    parser.add_argument(
        "--source",
        type=Path,
        default=Path("artifacts/strong_policy_upgrade/causal_expert_demonstrations_systematic"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/strong_policy_upgrade/hierarchical_demonstrations"),
    )
    parser.add_argument("--decision-interval", type=int, default=4)
    parser.add_argument("--records-per-chunk", type=int, default=10_000)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest = HierarchicalDatasetBuilder(
        args.source,
        args.output,
        HierarchicalDerivationConfig(
            decision_interval=args.decision_interval,
            records_per_chunk=args.records_per_chunk,
        ),
    ).build()
    print(json.dumps(manifest, indent=2))
    print("HIERARCHICAL_DEMONSTRATIONS=PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
