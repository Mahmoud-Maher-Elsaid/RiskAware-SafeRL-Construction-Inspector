from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"JSONL file was not found: {path}")

    records: list[dict[str, Any]] = []

    with path.open("r", encoding="utf-8") as file:
        for line in file:
            stripped = line.strip()
            if stripped:
                records.append(json.loads(stripped))

    if not records:
        raise ValueError(f"JSONL file contains no records: {path}")

    return records


def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def flatten_diagnostics(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    for record in records:
        row: dict[str, Any] = {
            "timesteps": record["timesteps"],
            "total_actions": record["total_actions"],
            "transitions": record["transitions"],
            "completed_episodes": record["completed_episodes"],
        }
        row.update(
            {
                f"action_frequency_{name}": value
                for name, value in record["action_frequencies"].items()
            }
        )
        row.update(
            {f"mean_step_{name}": value for name, value in record["mean_step_costs"].items()}
        )

        for metric_name in (
            "episode_hazard_recall_mean",
            "episode_coverage_mean",
            "episode_success_mean",
        ):
            row[metric_name] = record.get(metric_name)

        rows.append(row)

    return rows


def plot_action_frequencies(rows: list[dict[str, Any]], output_path: Path) -> None:
    timesteps = [int(row["timesteps"]) for row in rows]
    action_columns = sorted(key for key in rows[0] if key.startswith("action_frequency_"))

    plt.figure(figsize=(10, 6))

    for column in action_columns:
        plt.plot(
            timesteps,
            [float(row[column]) for row in rows],
            label=column.removeprefix("action_frequency_"),
        )

    plt.xlabel("Environment timesteps")
    plt.ylabel("Action frequency")
    plt.title("PPO action frequencies by rollout")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def plot_safety_cost(rows: list[dict[str, Any]], output_path: Path) -> None:
    timesteps = [int(row["timesteps"]) for row in rows]

    plt.figure(figsize=(10, 6))
    plt.plot(
        timesteps,
        [float(row["mean_step_safety_cost"]) for row in rows],
    )
    plt.xlabel("Environment timesteps")
    plt.ylabel("Mean safety cost per transition")
    plt.title("PPO rollout safety cost")
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def plot_validation_history(
    records: list[dict[str, Any]],
    output_path: Path,
) -> None:
    timesteps = [int(record["timesteps"]) for record in records]

    plt.figure(figsize=(10, 6))
    plt.plot(
        timesteps,
        [float(record["metrics"]["hazard_recall"]["mean"]) for record in records],
        label="hazard_recall",
    )
    plt.plot(
        timesteps,
        [float(record["metrics"]["success"]["mean"]) for record in records],
        label="success",
    )
    plt.plot(
        timesteps,
        [float(record["metrics"]["safety_cost"]["mean"]) for record in records],
        label="safety_cost",
    )
    plt.xlabel("Environment timesteps")
    plt.ylabel("Validation metric")
    plt.title("Deterministic validation history")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def main() -> None:
    args = parse_args()
    run_directory = args.run_dir
    output_directory = args.output_dir if args.output_dir is not None else run_directory / "reports"
    output_directory.mkdir(parents=True, exist_ok=True)

    metadata_path = run_directory / "run_metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    diagnostics = read_jsonl(run_directory / "diagnostics" / "rollout_diagnostics.jsonl")
    validation_history = read_jsonl(run_directory / "evaluations" / "evaluation_history.jsonl")
    rows = flatten_diagnostics(diagnostics)

    write_rows(output_directory / "rollout_diagnostics.csv", rows)
    plot_action_frequencies(
        rows,
        output_directory / "action_frequencies.png",
    )
    plot_safety_cost(
        rows,
        output_directory / "safety_cost.png",
    )
    plot_validation_history(
        validation_history,
        output_directory / "validation_history.png",
    )

    best_validation = max(
        validation_history,
        key=lambda record: float(record["selection_score"]),
    )
    summary = {
        "run_name": metadata["run_name"],
        "rollout_records": len(diagnostics),
        "first_rollout": diagnostics[0],
        "last_rollout": diagnostics[-1],
        "best_validation": best_validation,
        "report_directory": str(output_directory),
    }

    for name in (
        "final_feasible_validation",
        "final_full_validation",
    ):
        path = run_directory / "evaluations" / f"{name}_summary.json"
        if path.is_file():
            summary[name] = json.loads(path.read_text(encoding="utf-8"))

    summary_path = output_directory / "training_summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))
    print(f"Training report: {summary_path}")


if __name__ == "__main__":
    main()
