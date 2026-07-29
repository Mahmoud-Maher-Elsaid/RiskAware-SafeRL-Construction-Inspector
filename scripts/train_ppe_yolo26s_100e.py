from __future__ import annotations

import argparse
import csv
import gc
import importlib.metadata
import json
import platform
import statistics
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch
import yaml
from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"

if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from riskaware_saferrl.perception_dataset import (  # noqa: E402
    EXPECTED_DATASET_CLASS_NAMES,
    load_data_yaml,
    normalize_class_names,
    resolve_ppe_dataset_root,
    sha256_file,
    validate_dataset_layout,
)

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp",
    ".tif",
    ".tiff",
}


def load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Required JSON file was not found: {path}")

    payload = json.loads(path.read_text(encoding="utf-8"))

    if not isinstance(payload, dict):
        raise TypeError(f"Expected a JSON object in: {path}")

    return payload


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def package_version(distribution_name: str) -> str | None:
    try:
        return importlib.metadata.version(distribution_name)
    except importlib.metadata.PackageNotFoundError:
        return None


def scalar_value(value: Any) -> float | int | str | bool | None:
    if value is None:
        return None

    if isinstance(value, (bool, int, float, str)):
        return value

    if hasattr(value, "item"):
        try:
            item = value.item()

            if isinstance(item, (bool, int, float, str)):
                return item
        except (RuntimeError, TypeError, ValueError):
            pass

    try:
        return float(value)
    except (TypeError, ValueError):
        return str(value)


def to_list(value: Any) -> list[Any]:
    if value is None:
        return []

    if hasattr(value, "detach"):
        value = value.detach()

    if hasattr(value, "cpu"):
        value = value.cpu()

    if hasattr(value, "numpy"):
        value = value.numpy()

    if hasattr(value, "tolist"):
        converted = value.tolist()

        if isinstance(converted, list):
            return converted

        return [converted]

    if isinstance(value, (list, tuple)):
        return list(value)

    return [value]


def percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None

    ordered = sorted(values)

    if len(ordered) == 1:
        return ordered[0]

    position = (len(ordered) - 1) * fraction
    lower_index = int(position)
    upper_index = min(lower_index + 1, len(ordered) - 1)
    interpolation = position - lower_index

    return ordered[lower_index] * (1.0 - interpolation) + ordered[upper_index] * interpolation


def validate_configuration(config: dict[str, Any]) -> None:
    policy = config.get("training_policy", {})

    required_false_values = (
        "resume",
        "allow_pilot_checkpoints",
        "allow_historical_custom_ppe_checkpoint",
        "overwrite_existing_runs",
        "test_split_allowed",
        "final_production_model_selected_by_this_stage",
    )

    for key in required_false_values:
        if policy.get(key) is not False:
            raise ValueError(f"Training policy must set {key}=false.")

    if policy.get("allow_official_generic_pretrained_weights") is not True:
        raise ValueError("Official generic pretrained initialization must be enabled.")

    model_config = config.get("model", {})

    if model_config.get("name") != "yolo26s":
        raise ValueError("The production candidate must use YOLO26s.")

    training = config.get("training", {})

    if int(training.get("epochs", 0)) != 100:
        raise ValueError("Training must use exactly 100 epochs.")

    if int(training.get("image_size", 0)) != 640:
        raise ValueError("Training image size must be 640.")

    if float(training.get("fraction", 0.0)) != 1.0:
        raise ValueError("Training must use the complete training split.")

    if int(training.get("patience", 0)) < 100:
        raise ValueError("Early-stopping patience must not stop the 100-epoch run early.")


def validate_environment(config: dict[str, Any]) -> dict[str, Any]:
    lock = config["environment_lock"]

    actual = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "ultralytics": package_version("ultralytics"),
        "cuda_available": torch.cuda.is_available(),
    }

    expected_python = str(lock["python_major_minor"])

    if not actual["python"].startswith(expected_python + "."):
        raise RuntimeError(f"Expected Python {expected_python}.x, found {actual['python']}.")

    if actual["torch"] != lock["torch"]:
        raise RuntimeError(f"Expected PyTorch {lock['torch']}, found {actual['torch']}.")

    if actual["torch_cuda"] != lock["torch_cuda"]:
        raise RuntimeError(
            f"Expected CUDA build {lock['torch_cuda']}, found {actual['torch_cuda']}."
        )

    if actual["ultralytics"] != lock["ultralytics"]:
        raise RuntimeError(
            f"Expected Ultralytics {lock['ultralytics']}, found {actual['ultralytics']}."
        )

    if not actual["cuda_available"]:
        raise RuntimeError("CUDA is not available.")

    actual["gpu_name"] = torch.cuda.get_device_name(0)
    actual["gpu_total_memory_gib"] = round(
        torch.cuda.get_device_properties(0).total_memory / (1024**3),
        3,
    )

    return actual


def validate_project_prerequisites() -> None:
    integration_report = load_json(
        PROJECT_ROOT / "data" / "manifests" / "ppe_dataset_integration_validation.json"
    )

    environment_report = load_json(
        PROJECT_ROOT / "reports" / "perception" / "training_environment.json"
    )

    pilot_decision = load_json(
        PROJECT_ROOT / "reports" / "perception" / "pilot_runs" / "pilot_viability_decision.json"
    )

    if integration_report.get("status") != "PASS":
        raise RuntimeError("Dataset integration status is not PASS.")

    if environment_report.get("training_ready") is not True:
        raise RuntimeError("Training environment is not ready.")

    if pilot_decision.get("pilot_status") != "PASS":
        raise RuntimeError("Pilot viability status is not PASS.")

    if pilot_decision.get("final_production_model_selected") is not False:
        raise RuntimeError("The pilot incorrectly selected a production model.")

    priority = pilot_decision.get(
        "candidate_priority_for_full_dataset_screening",
        [],
    )

    if not priority or priority[0] != "yolo26s":
        raise RuntimeError("The pilot decision does not prioritize YOLO26s.")


def verify_generic_weight(config: dict[str, Any]) -> Path:
    model_config = config["model"]
    relative_path = Path(str(model_config["generic_weights"]))

    if relative_path.is_absolute():
        raise ValueError("Generic weight path must be project-relative.")

    weight_path = (PROJECT_ROOT / relative_path).resolve()

    allowed_directory = (
        PROJECT_ROOT / "artifacts" / "models" / "perception" / "generic_pretrained"
    ).resolve()

    try:
        weight_path.relative_to(allowed_directory)
    except ValueError as error:
        raise RuntimeError(
            f"Generic weight is outside the approved directory: {weight_path}"
        ) from error

    if not weight_path.is_file():
        raise FileNotFoundError(f"Generic pretrained weight was not found: {weight_path}")

    expected_hash = str(model_config["expected_sha256"]).lower()
    actual_hash = sha256_file(weight_path).lower()

    if actual_hash != expected_hash:
        raise RuntimeError(
            f"Generic weight SHA256 mismatch.\nExpected: {expected_hash}\nActual:   {actual_hash}"
        )

    return weight_path


def create_runtime_dataset(
    dataset_root: Path,
    output_directory: Path,
) -> tuple[Path, dict[str, Any]]:
    layout = validate_dataset_layout(dataset_root)
    source_yaml = load_data_yaml(layout.data_yaml)
    class_names = normalize_class_names(source_yaml.get("names"))

    if class_names != list(EXPECTED_DATASET_CLASS_NAMES):
        raise ValueError("Dataset class order does not match the locked schema.")

    output_directory.mkdir(parents=True, exist_ok=False)

    runtime_yaml = {
        "path": str(dataset_root),
        "train": str(layout.train_images),
        "val": str(layout.valid_images),
        "nc": len(class_names),
        "names": class_names,
    }

    runtime_yaml_path = output_directory / "data.yaml"

    runtime_yaml_path.write_text(
        yaml.safe_dump(
            runtime_yaml,
            sort_keys=False,
            allow_unicode=True,
        ),
        encoding="utf-8",
    )

    manifest = {
        "schema_version": 1,
        "created_at": datetime.now(UTC).isoformat(),
        "dataset_root": str(dataset_root),
        "source_data_yaml": str(layout.data_yaml),
        "source_data_yaml_sha256": sha256_file(layout.data_yaml),
        "runtime_data_yaml": str(runtime_yaml_path),
        "train_images": str(layout.train_images),
        "validation_images": str(layout.valid_images),
        "test_split_included": False,
        "class_names": class_names,
    }

    write_json(output_directory / "manifest.json", manifest)

    return runtime_yaml_path, manifest


def extract_overall_metrics(metrics: Any) -> dict[str, Any]:
    results_dict = getattr(metrics, "results_dict", {})

    if not isinstance(results_dict, dict):
        return {}

    return {str(key): scalar_value(value) for key, value in results_dict.items()}


def metric_value(
    metrics: dict[str, Any],
    keys: tuple[str, ...],
) -> float | None:
    for key in keys:
        value = metrics.get(key)

        if isinstance(value, (int, float)):
            return float(value)

    return None


def mean_ap_row(row: Any) -> float | None:
    values = to_list(row)

    numeric_values = [float(value) for value in values if isinstance(value, (int, float))]

    if not numeric_values:
        return None

    return statistics.fmean(numeric_values)


def extract_per_class_metrics(
    validation_results: Any,
) -> dict[str, dict[str, float | None]]:
    output = {
        class_name: {
            "precision": None,
            "recall": None,
            "map50": None,
            "map50_95": None,
        }
        for class_name in EXPECTED_DATASET_CLASS_NAMES
    }

    box_metrics = getattr(validation_results, "box", None)

    if box_metrics is None:
        return output

    class_indices = [int(value) for value in to_list(getattr(box_metrics, "ap_class_index", None))]

    precision_values = to_list(getattr(box_metrics, "p", None))
    recall_values = to_list(getattr(box_metrics, "r", None))
    map50_values = to_list(getattr(box_metrics, "ap50", None))
    ap_values = to_list(getattr(box_metrics, "ap", None))

    for position, class_id in enumerate(class_indices):
        if not 0 <= class_id < len(EXPECTED_DATASET_CLASS_NAMES):
            continue

        class_name = EXPECTED_DATASET_CLASS_NAMES[class_id]

        output[class_name] = {
            "precision": (
                float(precision_values[position]) if position < len(precision_values) else None
            ),
            "recall": (float(recall_values[position]) if position < len(recall_values) else None),
            "map50": (float(map50_values[position]) if position < len(map50_values) else None),
            "map50_95": (mean_ap_row(ap_values[position]) if position < len(ap_values) else None),
        }

    return output


def critical_class_mean_recall(
    per_class_metrics: dict[str, dict[str, float | None]],
    critical_classes: list[str],
) -> float | None:
    recall_values: list[float] = []

    for class_name in critical_classes:
        value = per_class_metrics.get(
            class_name,
            {},
        ).get("recall")

        if isinstance(value, (int, float)):
            recall_values.append(float(value))

    if not recall_values:
        return None

    return statistics.fmean(recall_values)


def write_class_metrics_csv(
    path: Path,
    metrics: dict[str, dict[str, float | None]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    rows = [
        {
            "class_id": class_id,
            "class_name": class_name,
            **metrics[class_name],
        }
        for class_id, class_name in enumerate(EXPECTED_DATASET_CLASS_NAMES)
    ]

    with path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as file_handle:
        writer = csv.DictWriter(
            file_handle,
            fieldnames=[
                "class_id",
                "class_name",
                "precision",
                "recall",
                "map50",
                "map50_95",
            ],
        )

        writer.writeheader()
        writer.writerows(rows)


def count_completed_epochs(results_csv: Path) -> int:
    if not results_csv.is_file():
        raise FileNotFoundError(f"Training results CSV was not found: {results_csv}")

    with results_csv.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as file_handle:
        return sum(1 for _ in csv.DictReader(file_handle))


def benchmark_model(
    model_path: Path,
    validation_images_directory: Path,
    config: dict[str, Any],
) -> dict[str, Any]:
    benchmark_config = config["benchmark"]
    training_config = config["training"]

    image_paths = sorted(
        path
        for path in validation_images_directory.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )

    requested_count = int(benchmark_config["image_count"])
    warmup_count = int(benchmark_config["warmup_count"])

    benchmark_images = image_paths[:requested_count]
    warmup_images = benchmark_images[:warmup_count]

    if len(benchmark_images) != requested_count:
        raise RuntimeError(
            f"Requested {requested_count} benchmark images, found {len(benchmark_images)}."
        )

    model = YOLO(str(model_path))

    for image_path in warmup_images:
        model.predict(
            source=str(image_path),
            imgsz=int(training_config["image_size"]),
            device=int(training_config["device"]),
            conf=float(benchmark_config["confidence"]),
            verbose=False,
        )

    torch.cuda.synchronize()

    latency_values: list[float] = []

    for image_path in benchmark_images:
        torch.cuda.synchronize()
        started_at = time.perf_counter()

        model.predict(
            source=str(image_path),
            imgsz=int(training_config["image_size"]),
            device=int(training_config["device"]),
            conf=float(benchmark_config["confidence"]),
            verbose=False,
        )

        torch.cuda.synchronize()

        latency_values.append((time.perf_counter() - started_at) * 1000.0)

    del model
    gc.collect()
    torch.cuda.empty_cache()

    return {
        "sample_count": len(latency_values),
        "precision_mode": "framework_default",
        "mean_ms": statistics.fmean(latency_values),
        "p50_ms": percentile(latency_values, 0.50),
        "p95_ms": percentile(latency_values, 0.95),
        "minimum_ms": min(latency_values),
        "maximum_ms": max(latency_values),
        "measurements_ms": latency_values,
    }


def main() -> int:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        type=Path,
        default=(PROJECT_ROOT / "configs" / "perception" / "production_candidate_100e.json"),
    )

    arguments = parser.parse_args()
    config = load_json(arguments.config.resolve())

    validate_configuration(config)
    validate_project_prerequisites()

    environment = validate_environment(config)
    weight_path = verify_generic_weight(config)

    dataset_root = resolve_ppe_dataset_root(project_root=PROJECT_ROOT)

    dataset_layout = validate_dataset_layout(dataset_root)

    timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")

    runtime_dataset_directory = (
        PROJECT_ROOT
        / "artifacts"
        / "perception"
        / "production_100e"
        / "runtime_datasets"
        / timestamp
    )

    run_root = PROJECT_ROOT / "artifacts" / "runs" / "perception_production_100e"

    reports_directory = PROJECT_ROOT / "reports" / "perception" / "production_training"

    run_name = f"yolo26s_100e_seed42_{timestamp}"
    expected_run_directory = run_root / run_name

    if expected_run_directory.exists():
        raise RuntimeError(f"Run directory already exists: {expected_run_directory}")

    runtime_data_yaml, runtime_manifest = create_runtime_dataset(
        dataset_root,
        runtime_dataset_directory,
    )

    active_run_report = {
        "schema_version": 1,
        "created_at": datetime.now(UTC).isoformat(),
        "status": "RUNNING",
        "run_name": run_name,
        "expected_run_directory": str(expected_run_directory),
        "generic_weight": str(weight_path),
        "generic_weight_sha256": sha256_file(weight_path),
        "runtime_data_yaml": str(runtime_data_yaml),
        "epochs_requested": 100,
        "test_split_used": False,
        "resume": False,
    }

    write_json(
        reports_directory / "active_run.json",
        active_run_report,
    )

    training_config = config["training"]

    print()
    print("=" * 72)
    print("STARTING YOLO26S FULL-DATASET 100-EPOCH TRAINING")
    print("=" * 72)
    print(f"Dataset: {dataset_root}")
    print(f"Generic weight: {weight_path}")
    print(f"Generic weight SHA256: {sha256_file(weight_path)}")
    print("Epochs: 100")
    print("Image size: 640")
    print("Dataset fraction: 1.0")
    print("Test split used: False")
    print("Resume: False")
    print("Pilot checkpoint used: False")
    print("Historical custom checkpoint used: False")
    print(f"Run directory: {expected_run_directory}")
    print("=" * 72)

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    started_at = time.perf_counter()
    model = YOLO(str(weight_path))

    try:
        training_results = model.train(
            data=str(runtime_data_yaml),
            epochs=int(training_config["epochs"]),
            imgsz=int(training_config["image_size"]),
            batch=float(training_config["batch_gpu_fraction"]),
            device=int(training_config["device"]),
            workers=int(training_config["workers"]),
            amp=bool(training_config["amp"]),
            cache=bool(training_config["cache"]),
            deterministic=bool(training_config["deterministic"]),
            seed=int(training_config["seed"]),
            optimizer=str(training_config["optimizer"]),
            patience=int(training_config["patience"]),
            save=bool(training_config["save"]),
            save_period=int(training_config["save_period"]),
            plots=bool(training_config["plots"]),
            val=bool(training_config["validation"]),
            fraction=float(training_config["fraction"]),
            cos_lr=bool(training_config["cosine_learning_rate"]),
            multi_scale=float(training_config["multi_scale"]),
            mosaic=float(training_config["mosaic"]),
            close_mosaic=int(training_config["close_mosaic"]),
            cls_pw=float(training_config["class_weight_power"]),
            pretrained=True,
            resume=False,
            project=str(run_root),
            name=run_name,
            exist_ok=False,
            verbose=True,
        )
    except Exception as error:
        failure_report = {
            **active_run_report,
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

    resolved_batch_size = scalar_value(
        getattr(
            getattr(model, "trainer", None),
            "batch_size",
            None,
        )
    )

    peak_allocated_gib = torch.cuda.max_memory_allocated() / (1024**3)

    peak_reserved_gib = torch.cuda.max_memory_reserved() / (1024**3)

    save_directory = Path(training_results.save_dir).resolve()
    best_model_path = save_directory / "weights" / "best.pt"
    last_model_path = save_directory / "weights" / "last.pt"
    results_csv_path = save_directory / "results.csv"

    for required_model in (
        best_model_path,
        last_model_path,
        results_csv_path,
    ):
        if not required_model.is_file():
            raise FileNotFoundError(f"Required training artifact was not created: {required_model}")

    completed_epochs = count_completed_epochs(results_csv_path)

    if completed_epochs != 100:
        raise RuntimeError(f"Expected 100 completed epochs, found {completed_epochs}.")

    del model
    gc.collect()
    torch.cuda.empty_cache()

    validation_config = config["final_validation"]
    validation_model = YOLO(str(best_model_path))

    validation_results = validation_model.val(
        data=str(runtime_data_yaml),
        split="val",
        imgsz=int(validation_config["image_size"]),
        batch=int(validation_config["batch_size"]),
        device=int(training_config["device"]),
        workers=int(validation_config["workers"]),
        plots=True,
        save_json=False,
        verbose=True,
        project=str(save_directory),
        name="final_full_validation",
        exist_ok=True,
    )

    overall_metrics = extract_overall_metrics(validation_results)
    per_class_metrics = extract_per_class_metrics(validation_results)

    del validation_model
    gc.collect()
    torch.cuda.empty_cache()

    benchmark = benchmark_model(
        best_model_path,
        dataset_layout.valid_images,
        config,
    )

    critical_classes = [str(value) for value in config["safety_priority_classes"]]

    class_metrics_csv = reports_directory / f"class_metrics_{timestamp}.csv"

    write_class_metrics_csv(
        class_metrics_csv,
        per_class_metrics,
    )

    summary = {
        "schema_version": 1,
        "created_at": datetime.now(UTC).isoformat(),
        "status": "PASS",
        "experiment_type": config["experiment_type"],
        "purpose": ("Create a strong 100-epoch full-dataset YOLO26s production candidate."),
        "environment": environment,
        "dataset_root": str(dataset_root),
        "runtime_dataset_manifest": runtime_manifest,
        "model_name": "yolo26s",
        "initialization_type": "official_generic_pretrained",
        "generic_weight": str(weight_path),
        "generic_weight_sha256": sha256_file(weight_path),
        "resume": False,
        "pilot_checkpoint_used": False,
        "historical_custom_checkpoint_used": False,
        "test_split_used": False,
        "epochs_requested": 100,
        "epochs_completed": completed_epochs,
        "requested_batch_gpu_fraction": float(training_config["batch_gpu_fraction"]),
        "resolved_batch_size": resolved_batch_size,
        "duration_seconds": duration_seconds,
        "peak_allocated_gib": peak_allocated_gib,
        "peak_reserved_gib": peak_reserved_gib,
        "training_config": training_config,
        "overall_metrics": overall_metrics,
        "precision": metric_value(
            overall_metrics,
            (
                "metrics/precision(B)",
                "metrics/precision",
            ),
        ),
        "recall": metric_value(
            overall_metrics,
            (
                "metrics/recall(B)",
                "metrics/recall",
            ),
        ),
        "map50": metric_value(
            overall_metrics,
            (
                "metrics/mAP50(B)",
                "metrics/mAP50",
            ),
        ),
        "map50_95": metric_value(
            overall_metrics,
            (
                "metrics/mAP50-95(B)",
                "metrics/mAP50-95",
            ),
        ),
        "per_class_metrics": per_class_metrics,
        "safety_priority_classes": critical_classes,
        "critical_class_mean_recall": (
            critical_class_mean_recall(
                per_class_metrics,
                critical_classes,
            )
        ),
        "benchmark": benchmark,
        "best_model": str(best_model_path),
        "best_model_sha256": sha256_file(best_model_path),
        "last_model": str(last_model_path),
        "last_model_sha256": sha256_file(last_model_path),
        "results_csv": str(results_csv_path),
        "class_metrics_csv": str(class_metrics_csv),
        "run_directory": str(save_directory),
        "production_candidate_created": True,
        "final_production_model_selected": False,
        "commit_created": False,
        "push_performed": False,
    }

    timestamped_summary_path = reports_directory / f"summary_{timestamp}.json"

    latest_summary_path = reports_directory / "latest_summary.json"

    write_json(timestamped_summary_path, summary)
    write_json(latest_summary_path, summary)

    active_run_report["status"] = "COMPLETED"
    active_run_report["updated_at"] = datetime.now(UTC).isoformat()
    active_run_report["summary"] = str(timestamped_summary_path)
    active_run_report["best_model"] = str(best_model_path)

    write_json(
        reports_directory / "active_run.json",
        active_run_report,
    )

    print()
    print("=" * 72)
    print("YOLO26S 100-EPOCH TRAINING SUMMARY")
    print("=" * 72)
    print("Status: PASS")
    print(f"Epochs completed: {completed_epochs}")
    print(f"Resolved batch size: {resolved_batch_size}")
    print(f"Precision: {summary['precision']}")
    print(f"Recall: {summary['recall']}")
    print(f"mAP50: {summary['map50']}")
    print(f"mAP50-95: {summary['map50_95']}")
    print(f"Critical-class mean recall: {summary['critical_class_mean_recall']}")
    print(f"Mean inference latency ms: {benchmark['mean_ms']}")
    print(f"Best model: {best_model_path}")
    print(f"Best SHA256: {summary['best_model_sha256']}")
    print(f"Summary: {timestamped_summary_path}")
    print(f"Latest summary: {latest_summary_path}")
    print("=" * 72)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
