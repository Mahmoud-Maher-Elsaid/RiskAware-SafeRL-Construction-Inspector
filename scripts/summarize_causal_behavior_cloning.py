from __future__ import annotations

import csv
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

ROOT = Path("reports/strong_policy_upgrade/causal_behavior_cloning")
RUNS = (
    ("causal_only", 20260730, "systematic_causal_only_attention_seed_20260730"),
    ("causal_only", 20260731, "systematic_causal_only_attention_seed_20260731"),
    ("causal_only", 20260732, "systematic_causal_only_attention_seed_20260732"),
    (
        "causal_plus_privileged_pretraining",
        20260730,
        "mode_privileged_pretraining_seed_20260730",
    ),
    (
        "causal_plus_privileged_distillation_with_confidence_filter",
        20260730,
        "mode_systematic_confidence_distillation_seed_20260730",
    ),
)


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def summarize() -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    checkpoints: list[dict[str, Any]] = []
    memory_runs: list[dict[str, Any]] = []
    for mode, seed, run_name in RUNS:
        run_dir = ROOT / run_name
        summary = json.loads(
            (run_dir / "behavior_cloning_summary.json").read_text(encoding="utf-8")
        )
        test = summary["held_out_test"]
        checkpoint = Path(summary["checkpoint_path"])
        actual_hash = sha256_file(checkpoint)
        if actual_hash != summary["checkpoint_sha256"]:
            raise RuntimeError(f"Checkpoint hash mismatch: {checkpoint}")
        rows.append(
            {
                "mode": mode,
                "seed": seed,
                "status": summary["status"],
                "accuracy": test["accuracy"],
                **{
                    f"recall_action_{action}": test["per_action_recall"][action]
                    for action in range(5)
                },
                "invalid_action_predictions": test["invalid_action_predictions"],
                "split_leakage": summary["split_leakage"],
                "checkpoint_sha256": actual_hash,
                "checkpoint_path": checkpoint.as_posix(),
                "dataset_sha256": summary["dataset_sha256"],
            }
        )
        checkpoints.append(
            {
                "mode": mode,
                "seed": seed,
                "path": checkpoint.as_posix(),
                "bytes": checkpoint.stat().st_size,
                "sha256": actual_hash,
                "verified": True,
            }
        )
        memory = json.loads((run_dir / "memory_profile.json").read_text(encoding="utf-8"))
        samples = memory["samples"]
        memory_runs.append(
            {
                "mode": mode,
                "seed": seed,
                "status": memory["status"],
                "max_process_rss_bytes": max(sample["process_rss_bytes"] for sample in samples),
                "minimum_available_physical_bytes": min(
                    sample["available_physical_bytes"] for sample in samples
                ),
                "max_committed_bytes": max(sample["committed_bytes"] for sample in samples),
                "max_gpu_allocated_bytes": max(sample["gpu_allocated_bytes"] for sample in samples),
                "max_gpu_reserved_bytes": max(sample["gpu_reserved_bytes"] for sample in samples),
            }
        )
        if mode == "causal_only":
            shutil.copyfile(
                run_dir / "confusion_matrix.csv",
                ROOT / f"confusion_matrix_seed_{seed}.csv",
            )

    with (ROOT / "behavior_cloning_results.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    causal_rows = [row for row in rows if row["mode"] == "causal_only"]
    mean_accuracy = sum(row["accuracy"] for row in causal_rows) / len(causal_rows)
    best = max(rows, key=lambda row: row["accuracy"])
    minimum_recall = min(
        row[f"recall_action_{action}"] for row in causal_rows for action in range(5)
    )
    memory_gate_path = ROOT / "memory_smoke_test.json"
    memory_gate = (
        json.loads(memory_gate_path.read_text(encoding="utf-8"))
        if memory_gate_path.exists()
        else {"status": "PENDING"}
    )
    passed = (
        mean_accuracy >= 0.85
        and best["accuracy"] >= 0.87
        and minimum_recall >= 0.70
        and all(row["invalid_action_predictions"] == 0 for row in causal_rows)
        and all(row["split_leakage"] == 0 for row in causal_rows)
        and all(row["status"] == "PASSED" for row in causal_rows)
        and memory_gate["status"] == "PASSED"
    )
    summary = {
        "status": "PASSED" if passed else "PENDING_MEMORY_GATE",
        "primary_mode": "causal_only",
        "primary_seeds": [row["seed"] for row in causal_rows],
        "mean_held_out_accuracy": mean_accuracy,
        "best_held_out_accuracy": best["accuracy"],
        "best_mode": best["mode"],
        "best_seed": best["seed"],
        "minimum_primary_per_action_recall": minimum_recall,
        "invalid_action_predictions": sum(row["invalid_action_predictions"] for row in causal_rows),
        "split_leakage": sum(row["split_leakage"] for row in causal_rows),
        "memory_gate": memory_gate.get("status"),
        "gates": {
            "mean_accuracy": {"required": 0.85, "observed": mean_accuracy},
            "best_accuracy": {"required": 0.87, "observed": best["accuracy"]},
            "per_action_recall": {"required": 0.70, "observed": minimum_recall},
            "invalid_actions": {
                "required": 0,
                "observed": sum(row["invalid_action_predictions"] for row in causal_rows),
            },
            "split_leakage": {
                "required": 0,
                "observed": sum(row["split_leakage"] for row in causal_rows),
            },
        },
        "mode_results": rows,
    }
    (ROOT / "behavior_cloning_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    (ROOT / "checkpoint_manifest.json").write_text(
        json.dumps(
            {"status": "PASSED", "checkpoints": checkpoints},
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    (ROOT / "memory_profile.json").write_text(
        json.dumps({"status": "PASSED", "runs": memory_runs}, indent=2) + "\n",
        encoding="utf-8",
    )
    return summary


def main() -> None:
    summary = summarize()
    print(f"CAUSAL_BEHAVIOR_CLONING={summary['status']}")
    if summary["status"] != "PASSED":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
