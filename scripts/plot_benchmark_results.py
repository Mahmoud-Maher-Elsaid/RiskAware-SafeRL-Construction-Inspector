from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("summary", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    summary = json.loads(arguments.summary.read_text(encoding="utf-8"))
    methods = sorted(summary)
    reward = [summary[name]["metrics"]["reward"]["mean"] for name in methods]
    cost = [summary[name]["metrics"]["safety_cost"]["mean"] for name in methods]
    recall = [summary[name]["metrics"]["hazard_recall"]["mean"] for name in methods]

    figure, axes = plt.subplots(1, 3, figsize=(15, 5))
    for axis, values, title in zip(
        axes,
        (reward, cost, recall),
        ("Mean reward", "Mean safety cost", "Mean hazard recall"),
        strict=True,
    ):
        axis.bar(methods, values)
        axis.set_title(title)
        axis.tick_params(axis="x", rotation=65)
        axis.grid(axis="y", alpha=0.25)
    figure.suptitle("Validation benchmark (generated from JSON)")
    figure.tight_layout()
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(arguments.output, dpi=180)
    plt.close(figure)


if __name__ == "__main__":
    main()
