from __future__ import annotations

import csv
import hashlib
import json
import platform
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch
import ultralytics
from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATASET_YAML = (
    PROJECT_ROOT / "Personal Protective Equipment - Combined Model.v1i.yolov8" / "data.yaml"
)

GENERIC_WEIGHT = (
    PROJECT_ROOT / "artifacts" / "models" / "perception" / "generic_pretrained" / "yolo26s.pt"
)

RUNS_ROOT = PROJECT_ROOT / "artifacts" / "runs" / "perception_production_optimized_100e"

REPORTS_ROOT = PROJECT_ROOT / "reports" / "perception" / "production_training_optimized"

EXPECTED_CLASSES = [
    "Fall-Detected",
    "Gloves",
    "Goggles",
    "Hardhat",
    "Ladder",
    "Mask",
    "NO-Gloves",
    "NO-Goggles",
    "NO-Hardhat",
    "NO-Mask",
    "NO-Safety Vest",
    "Person",
    "Safety Cone",
    "Safety Vest",
]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as file_handle:
        while chunk := file_handle.read(1024 * 1024):
            digest.update(chunk)

    return digest.hexdigest()


def write_json(
    path: Path,
    payload: dict[str, Any],
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    path.write_text(
        json.dumps(
            payload,
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )


def count_completed_epochs(
    results_csv: Path,
) -> int:
    if not results_csv.is_file():
        raise FileNotFoundError(f"Results CSV was not found: {results_csv}")

    with results_csv.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as file_handle:
        return sum(1 for _ in csv.DictReader(file_handle))


def metric_value(
    metrics: dict[str, Any],
    names: tuple[str, ...],
) -> float | None:
    for name in names:
        value = metrics.get(name)

        if value is None:
            continue

        if hasattr(value, "item"):
            value = value.item()

        try:
            return float(value)
        except (TypeError, ValueError):
            continue

    return None


def validate_environment() -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available in the repository environment.")

    if not DATASET_YAML.is_file():
        raise FileNotFoundError(f"Dataset YAML was not found: {DATASET_YAML}")

    if not GENERIC_WEIGHT.is_file():
        raise FileNotFoundError(f"Generic YOLO26s weight was not found: {GENERIC_WEIGHT}")

    properties = torch.cuda.get_device_properties(0)

    return {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "ultralytics": ultralytics.__version__,
        "gpu_name": torch.cuda.get_device_name(0),
        "gpu_total_memory_gib": round(
            properties.total_memory / (1024**3),
            3,
        ),
    }


def configure_gpu_performance() -> None:
    torch.cuda.empty_cache()

    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    torch.set_float32_matmul_precision("high")


def main() -> int:
    environment = validate_environment()
    configure_gpu_performance()

    timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")

    run_name = f"yolo26s_gpu90_fixed640_100e_seed42_{timestamp}"

    expected_run_directory = RUNS_ROOT / run_name

    if expected_run_directory.exists():
        raise RuntimeError(f"The target run directory already exists: {expected_run_directory}")

    generic_weight_hash = sha256_file(GENERIC_WEIGHT)

    active_report = {
        "schema_version": 1,
        "created_at": datetime.now(UTC).isoformat(),
        "status": "RUNNING",
        "model": "yolo26s",
        "initialization": "official_generic_pretrained",
        "generic_weight": str(GENERIC_WEIGHT),
        "generic_weight_sha256": generic_weight_hash,
        "dataset_yaml": str(DATASET_YAML),
        "epochs_requested": 100,
        "image_size": 640,
        "batch_gpu_memory_fraction": 0.90,
        "multi_scale": False,
        "amp": True,
        "test_split_used": False,
        "resume": False,
        "expected_run_directory": str(expected_run_directory),
        "environment": environment,
    }

    write_json(
        REPORTS_ROOT / "active_run.json",
        active_report,
    )

    print()
    print("=" * 72)
    print("OPTIMIZED YOLO26S PPE TRAINING")
    print("=" * 72)
    print(f"Python: {environment['python']}")
    print(f"PyTorch: {environment['torch']}")
    print(f"CUDA build: {environment['torch_cuda']}")
    print(f"Ultralytics: {environment['ultralytics']}")
    print(f"GPU: {environment['gpu_name']}")
    print(f"GPU memory GiB: {environment['gpu_total_memory_gib']}")
    print(f"Generic weight: {GENERIC_WEIGHT}")
    print(f"Generic weight SHA256: {generic_weight_hash}")
    print(f"Dataset: {DATASET_YAML}")
    print("Epochs: 100")
    print("Image size: 640 fixed")
    print("GPU memory target: 90 percent")
    print("AMP: enabled")
    print("Workers: 8")
    print("Disk cache: enabled")
    print("Multi-scale: disabled")
    print("Optimizer: automatic MuSGD selection")
    print("Deterministic mode: disabled for speed")
    print("Test split used: False")
    print("Resume: False")
    print(f"Run directory: {expected_run_directory}")
    print("=" * 72)
    print()

    model = YOLO(str(GENERIC_WEIGHT))

    started_at = time.perf_counter()

    try:
        training_results = model.train(
            data=str(DATASET_YAML),
            epochs=100,
            imgsz=640,
            batch=0.90,
            device=0,
            workers=8,
            cache="disk",
            amp=True,
            optimizer="auto",
            patience=100,
            seed=42,
            deterministic=False,
            rect=False,
            multi_scale=0.0,
            cos_lr=True,
            warmup_epochs=3.0,
            mosaic=1.0,
            close_mosaic=10,
            fraction=1.0,
            save=True,
            save_period=5,
            plots=True,
            val=True,
            pretrained=True,
            resume=False,
            profile=False,
            project=str(RUNS_ROOT),
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
            "elapsed_seconds": (time.perf_counter() - started_at),
        }

        write_json(
            REPORTS_ROOT / "latest_failure.json",
            failure_report,
        )

        raise

    duration_seconds = time.perf_counter() - started_at

    save_directory = Path(training_results.save_dir).resolve()

    best_model = save_directory / "weights" / "best.pt"

    last_model = save_directory / "weights" / "last.pt"

    results_csv = save_directory / "results.csv"

    for required_path in (
        best_model,
        last_model,
        results_csv,
    ):
        if not required_path.is_file():
            raise FileNotFoundError(f"Required training artifact was not created: {required_path}")

    completed_epochs = count_completed_epochs(results_csv)

    if completed_epochs != 100:
        raise RuntimeError(
            f"Training did not complete exactly 100 epochs. Found: {completed_epochs}"
        )

    resolved_batch_size = getattr(
        getattr(model, "trainer", None),
        "batch_size",
        None,
    )

    peak_allocated_gib = round(
        torch.cuda.max_memory_allocated() / (1024**3),
        3,
    )

    peak_reserved_gib = round(
        torch.cuda.max_memory_reserved() / (1024**3),
        3,
    )

    del model
    torch.cuda.empty_cache()

    print()
    print("Running final validation on best.pt...")

    validation_model = YOLO(str(best_model))

    validation_results = validation_model.val(
        data=str(DATASET_YAML),
        split="val",
        imgsz=640,
        batch=8,
        device=0,
        workers=8,
        plots=True,
        save_json=False,
        project=str(save_directory),
        name="final_best_validation",
        exist_ok=True,
        verbose=True,
    )

    raw_metrics = getattr(
        validation_results,
        "results_dict",
        {},
    )

    if not isinstance(raw_metrics, dict):
        raw_metrics = {}

    summary = {
        "schema_version": 1,
        "created_at": datetime.now(UTC).isoformat(),
        "status": "PASS",
        "model": "yolo26s",
        "initialization": ("official_generic_pretrained"),
        "generic_weight": str(GENERIC_WEIGHT),
        "generic_weight_sha256": (generic_weight_hash),
        "dataset_yaml": str(DATASET_YAML),
        "epochs_requested": 100,
        "epochs_completed": completed_epochs,
        "image_size": 640,
        "batch_gpu_memory_fraction": 0.90,
        "resolved_batch_size": (resolved_batch_size),
        "multi_scale": False,
        "amp": True,
        "optimizer": "auto",
        "duration_seconds": duration_seconds,
        "peak_allocated_gib": (peak_allocated_gib),
        "peak_reserved_gib": (peak_reserved_gib),
        "precision": metric_value(
            raw_metrics,
            (
                "metrics/precision(B)",
                "metrics/precision",
            ),
        ),
        "recall": metric_value(
            raw_metrics,
            (
                "metrics/recall(B)",
                "metrics/recall",
            ),
        ),
        "map50": metric_value(
            raw_metrics,
            (
                "metrics/mAP50(B)",
                "metrics/mAP50",
            ),
        ),
        "map50_95": metric_value(
            raw_metrics,
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
        "run_directory": str(save_directory),
        "test_split_used": False,
        "resume": False,
        "environment": environment,
    }

    timestamped_summary = REPORTS_ROOT / f"summary_{timestamp}.json"

    latest_summary = REPORTS_ROOT / "latest_summary.json"

    write_json(
        timestamped_summary,
        summary,
    )

    write_json(
        latest_summary,
        summary,
    )

    active_report.update(
        {
            "updated_at": datetime.now(UTC).isoformat(),
            "status": "COMPLETED",
            "run_directory": str(save_directory),
            "best_model": str(best_model),
            "summary": str(timestamped_summary),
        }
    )

    write_json(
        REPORTS_ROOT / "active_run.json",
        active_report,
    )

    print()
    print("=" * 72)
    print("OPTIMIZED 100-EPOCH TRAINING SUMMARY")
    print("=" * 72)
    print("Status: PASS")
    print(f"Epochs completed: {completed_epochs}")
    print(f"Resolved batch size: {resolved_batch_size}")
    print(f"Peak allocated GiB: {peak_allocated_gib}")
    print(f"Peak reserved GiB: {peak_reserved_gib}")
    print(f"Precision: {summary['precision']}")
    print(f"Recall: {summary['recall']}")
    print(f"mAP50: {summary['map50']}")
    print(f"mAP50-95: {summary['map50_95']}")
    print(f"Best model: {best_model}")
    print(f"Best SHA256: {summary['best_model_sha256']}")
    print(f"Latest summary: {latest_summary}")
    print("=" * 72)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
