from __future__ import annotations

import argparse
import csv
import hashlib
import json
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Configuration was not found: {path}")

    payload = json.loads(path.read_text(encoding="utf-8"))

    if not isinstance(payload, dict):
        raise TypeError("Configuration must contain a JSON object.")

    return payload


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as file_handle:
        while chunk := file_handle.read(1024 * 1024):
            digest.update(chunk)

    return digest.hexdigest()


def read_results(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(f"Results CSV was not found: {path}")

    with path.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as file_handle:
        return list(csv.DictReader(file_handle))


def find_metric(
    row: dict[str, str],
    candidates: tuple[str, ...],
) -> float | None:
    normalized = {key.strip(): value for key, value in row.items()}

    for candidate in candidates:
        value = normalized.get(candidate)

        if value is None or value == "":
            continue

        try:
            return float(value)
        except ValueError:
            continue

    return None


def main() -> int:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        type=Path,
        required=True,
    )

    arguments = parser.parse_args()
    config = load_json(arguments.config.resolve())

    checkpoint_path = Path(config["source_checkpoint"]).resolve()
    runtime_data_yaml = Path(config["runtime_data_yaml"]).resolve()

    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Source checkpoint was not found: {checkpoint_path}")

    if not runtime_data_yaml.is_file():
        raise FileNotFoundError(f"Runtime dataset YAML was not found: {runtime_data_yaml}")

    expected_hash = str(config["source_checkpoint_sha256"]).lower()

    actual_hash = sha256_file(checkpoint_path).lower()

    if actual_hash != expected_hash:
        raise RuntimeError(
            "Source checkpoint SHA256 validation failed.\n"
            f"Expected: {expected_hash}\n"
            f"Actual:   {actual_hash}"
        )

    training = config["training"]
    timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")

    run_root = PROJECT_ROOT / "artifacts" / "runs" / "perception_fast_balanced"

    run_name = f"yolo26s_fast512_35e_seed42_{timestamp}"

    reports_directory = PROJECT_ROOT / "reports" / "perception" / "fast_balanced_training"

    active_report = {
        "schema_version": 1,
        "created_at": datetime.now(UTC).isoformat(),
        "status": "RUNNING",
        "source_checkpoint": str(checkpoint_path),
        "source_checkpoint_sha256": actual_hash,
        "runtime_data_yaml": str(runtime_data_yaml),
        "run_name": run_name,
        "expected_run_directory": str(run_root / run_name),
        "epochs_requested": int(training["epochs"]),
        "image_size": int(training["image_size"]),
        "batch": int(training["batch"]),
        "test_split_used": False,
        "resume_original_run": False,
    }

    write_json(
        reports_directory / "active_run.json",
        active_report,
    )

    print()
    print("=" * 72)
    print("STARTING FAST BALANCED YOLO26S TRAINING")
    print("=" * 72)
    print(f"Source checkpoint: {checkpoint_path}")
    print(f"Checkpoint SHA256: {actual_hash}")
    print(f"Runtime dataset: {runtime_data_yaml}")
    print(f"Epochs: {training['epochs']}")
    print(f"Image size: {training['image_size']}")
    print("Batch mode: automatic")
    print("Multi-scale: disabled")
    print(f"Early stopping patience: {training['patience']}")
    print("Test split used: False")
    print("Original run resume: False")
    print("=" * 72)

    model = YOLO(str(checkpoint_path))
    started_at = time.perf_counter()

    try:
        training_results = model.train(
            data=str(runtime_data_yaml),
            epochs=int(training["epochs"]),
            imgsz=int(training["image_size"]),
            batch=int(training["batch"]),
            device=int(training["device"]),
            workers=int(training["workers"]),
            amp=bool(training["amp"]),
            cache=bool(training["cache"]),
            deterministic=bool(training["deterministic"]),
            seed=int(training["seed"]),
            optimizer=str(training["optimizer"]),
            patience=int(training["patience"]),
            cos_lr=bool(training["cosine_learning_rate"]),
            multi_scale=float(training["multi_scale"]),
            mosaic=float(training["mosaic"]),
            close_mosaic=int(training["close_mosaic"]),
            save=True,
            save_period=int(training["save_period"]),
            plots=True,
            val=bool(training["validation"]),
            fraction=float(training["fraction"]),
            resume=False,
            pretrained=True,
            project=str(run_root),
            name=run_name,
            exist_ok=False,
            verbose=True,
        )
    except Exception as error:
        failure_report = {
            **active_report,
            "updated_at": datetime.now(UTC).isoformat(),
            "status": "FAIL",
            "error_type": type(error).__name__,
            "error": str(error),
            "elapsed_seconds": time.perf_counter() - started_at,
        }

        write_json(
            reports_directory / "latest_failure.json",
            failure_report,
        )

        raise

    duration_seconds = time.perf_counter() - started_at
    run_directory = Path(training_results.save_dir).resolve()

    best_model = run_directory / "weights" / "best.pt"
    last_model = run_directory / "weights" / "last.pt"
    results_csv = run_directory / "results.csv"

    for required_path in (
        best_model,
        last_model,
        results_csv,
    ):
        if not required_path.is_file():
            raise FileNotFoundError(f"Required training artifact was not created: {required_path}")

    result_rows = read_results(results_csv)

    if not result_rows:
        raise RuntimeError("Training results CSV contains no completed epochs.")

    final_row = result_rows[-1]

    summary = {
        "schema_version": 1,
        "created_at": datetime.now(UTC).isoformat(),
        "status": "PASS",
        "experiment_type": config["experiment_type"],
        "source_checkpoint": str(checkpoint_path),
        "source_checkpoint_sha256": actual_hash,
        "epochs_requested": int(training["epochs"]),
        "epochs_completed": len(result_rows),
        "image_size": int(training["image_size"]),
        "batch_mode": "automatic",
        "multi_scale": False,
        "test_split_used": False,
        "resume_original_run": False,
        "duration_seconds": duration_seconds,
        "precision": find_metric(
            final_row,
            (
                "metrics/precision(B)",
                "metrics/precision",
            ),
        ),
        "recall": find_metric(
            final_row,
            (
                "metrics/recall(B)",
                "metrics/recall",
            ),
        ),
        "map50": find_metric(
            final_row,
            (
                "metrics/mAP50(B)",
                "metrics/mAP50",
            ),
        ),
        "map50_95": find_metric(
            final_row,
            (
                "metrics/mAP50-95(B)",
                "metrics/mAP50-95",
            ),
        ),
        "best_model": str(best_model),
        "best_model_sha256": sha256_file(best_model),
        "last_model": str(last_model),
        "last_model_sha256": sha256_file(last_model),
        "results_csv": str(results_csv),
        "run_directory": str(run_directory),
        "final_result_row": final_row,
    }

    write_json(
        reports_directory / f"summary_{timestamp}.json",
        summary,
    )

    write_json(
        reports_directory / "latest_summary.json",
        summary,
    )

    active_report["status"] = "COMPLETED"
    active_report["updated_at"] = datetime.now(UTC).isoformat()
    active_report["run_directory"] = str(run_directory)
    active_report["best_model"] = str(best_model)

    write_json(
        reports_directory / "active_run.json",
        active_report,
    )

    print()
    print("=" * 72)
    print("FAST BALANCED TRAINING SUMMARY")
    print("=" * 72)
    print("Status: PASS")
    print(f"Epochs completed: {len(result_rows)}")
    print(f"Precision: {summary['precision']}")
    print(f"Recall: {summary['recall']}")
    print(f"mAP50: {summary['map50']}")
    print(f"mAP50-95: {summary['map50_95']}")
    print(f"Best model: {best_model}")
    print(f"Best SHA256: {summary['best_model_sha256']}")
    print(f"Run directory: {run_directory}")
    print("=" * 72)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
