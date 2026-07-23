from __future__ import annotations

import argparse
import csv
import gc
import importlib.metadata
import json
import os
import random
import statistics
import sys
import time
from collections import Counter, defaultdict
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

ALLOWED_GENERIC_WEIGHTS = {
    "yolo26n.pt",
    "yolo26s.pt",
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


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "candidate",
        "status",
        "batch_size",
        "duration_seconds",
        "peak_allocated_gib",
        "peak_reserved_gib",
        "metrics_precision",
        "metrics_recall",
        "metrics_map50",
        "metrics_map50_95",
        "latency_mean_ms",
        "latency_p50_ms",
        "latency_p95_ms",
        "best_model",
        "best_model_sha256",
        "run_directory",
        "error",
    ]

    with path.open("w", encoding="utf-8", newline="") as file_handle:
        writer = csv.DictWriter(
            file_handle,
            fieldnames=fieldnames,
            extrasaction="ignore",
        )

        writer.writeheader()
        writer.writerows(rows)


def package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def percentile(values: list[float], percentile_value: float) -> float | None:
    if not values:
        return None

    ordered = sorted(values)

    if len(ordered) == 1:
        return ordered[0]

    position = (len(ordered) - 1) * percentile_value
    lower_index = int(position)
    upper_index = min(lower_index + 1, len(ordered) - 1)
    fraction = position - lower_index

    return ordered[lower_index] * (1.0 - fraction) + ordered[upper_index] * fraction


def scalar_value(value: Any) -> float | int | str | bool | None:
    if value is None:
        return None

    if isinstance(value, bool | int | float | str):
        return value

    if hasattr(value, "item"):
        try:
            item = value.item()

            if isinstance(item, bool | int | float | str):
                return item
        except (RuntimeError, TypeError, ValueError):
            pass

    try:
        return float(value)
    except (TypeError, ValueError):
        return str(value)


def parse_label_classes(label_path: Path) -> set[int]:
    class_ids: set[int] = set()

    for line_number, raw_line in enumerate(
        label_path.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        line = raw_line.strip()

        if not line:
            continue

        values = line.split()

        if len(values) != 5:
            raise ValueError(f"Invalid label row in {label_path} at line {line_number}.")

        class_value = float(values[0])
        class_id = int(class_value)

        if class_value != class_id:
            raise ValueError(f"Non-integer class ID in {label_path} at line {line_number}.")

        if not 0 <= class_id < len(EXPECTED_DATASET_CLASS_NAMES):
            raise ValueError(f"Class ID {class_id} is outside the expected range.")

        class_ids.add(class_id)

    return class_ids


def build_split_records(
    images_directory: Path,
    labels_directory: Path,
) -> list[dict[str, Any]]:
    images_by_key: dict[str, Path] = {}

    for image_path in images_directory.rglob("*"):
        if not image_path.is_file() or image_path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue

        key = image_path.relative_to(images_directory).with_suffix("").as_posix().lower()

        if key in images_by_key:
            raise RuntimeError(f"Duplicate image key was found: {key}")

        images_by_key[key] = image_path.resolve()

    records: list[dict[str, Any]] = []

    for label_path in sorted(labels_directory.rglob("*.txt")):
        key = label_path.relative_to(labels_directory).with_suffix("").as_posix().lower()

        image_path = images_by_key.get(key)

        if image_path is None:
            raise RuntimeError(f"No image was found for label file: {label_path}")

        records.append(
            {
                "key": key,
                "image_path": image_path,
                "label_path": label_path.resolve(),
                "class_ids": parse_label_classes(label_path),
            }
        )

    if len(records) != len(images_by_key):
        raise RuntimeError("Image and label counts differ while building the pilot subset.")

    return records


def select_stratified_subset(
    records: list[dict[str, Any]],
    target_count: int,
    minimum_per_class: int,
    seed: int,
) -> list[dict[str, Any]]:
    if target_count <= 0:
        raise ValueError("Target subset count must be positive.")

    if target_count > len(records):
        raise ValueError(f"Requested {target_count} records but only {len(records)} exist.")

    random_generator = random.Random(seed)

    indices_by_class: defaultdict[int, list[int]] = defaultdict(list)

    for record_index, record in enumerate(records):
        for class_id in record["class_ids"]:
            indices_by_class[class_id].append(record_index)

    selected_indices: set[int] = set()

    class_order = sorted(
        range(len(EXPECTED_DATASET_CLASS_NAMES)),
        key=lambda class_id: len(indices_by_class[class_id]),
    )

    for class_id in class_order:
        candidate_indices = list(indices_by_class[class_id])
        random_generator.shuffle(candidate_indices)

        selected_for_class = 0

        for record_index in candidate_indices:
            if selected_for_class >= minimum_per_class:
                break

            if record_index in selected_indices:
                selected_for_class += 1
                continue

            selected_indices.add(record_index)
            selected_for_class += 1

    remaining_indices = [index for index in range(len(records)) if index not in selected_indices]

    random_generator.shuffle(remaining_indices)

    for record_index in remaining_indices:
        if len(selected_indices) >= target_count:
            break

        selected_indices.add(record_index)

    if len(selected_indices) < target_count:
        raise RuntimeError(f"Could only select {len(selected_indices)} of {target_count} records.")

    if len(selected_indices) > target_count:
        selected_list = list(selected_indices)
        random_generator.shuffle(selected_list)
        selected_indices = set(selected_list[:target_count])

    selected_records = [records[index] for index in sorted(selected_indices)]

    selected_class_counts: Counter[int] = Counter()

    for record in selected_records:
        for class_id in record["class_ids"]:
            selected_class_counts[class_id] += 1

    missing_classes = [
        EXPECTED_DATASET_CLASS_NAMES[class_id]
        for class_id in range(len(EXPECTED_DATASET_CLASS_NAMES))
        if selected_class_counts[class_id] == 0
    ]

    if missing_classes:
        raise RuntimeError(
            "The pilot subset does not contain all classes: " + ", ".join(missing_classes)
        )

    return selected_records


def write_image_list(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    content = "\n".join(record["image_path"].as_posix() for record in records)

    path.write_text(content + "\n", encoding="utf-8")


def class_image_counts(records: list[dict[str, Any]]) -> dict[str, int]:
    counts: Counter[int] = Counter()

    for record in records:
        for class_id in record["class_ids"]:
            counts[class_id] += 1

    return {
        EXPECTED_DATASET_CLASS_NAMES[class_id]: counts[class_id]
        for class_id in range(len(EXPECTED_DATASET_CLASS_NAMES))
    }


def create_pilot_dataset(
    dataset_root: Path,
    config: dict[str, Any],
    output_directory: Path,
) -> tuple[Path, dict[str, Any], list[Path]]:
    layout = validate_dataset_layout(dataset_root)

    original_yaml = load_data_yaml(layout.data_yaml)
    class_names = normalize_class_names(original_yaml.get("names"))

    if class_names != list(EXPECTED_DATASET_CLASS_NAMES):
        raise ValueError("Dataset class order does not match the locked schema.")

    subset_config = config["subset"]

    train_records = build_split_records(
        layout.train_images,
        layout.train_labels,
    )

    validation_records = build_split_records(
        layout.valid_images,
        layout.valid_labels,
    )

    selected_train = select_stratified_subset(
        train_records,
        int(subset_config["train_images"]),
        int(subset_config["minimum_train_images_per_class"]),
        int(subset_config["seed"]),
    )

    selected_validation = select_stratified_subset(
        validation_records,
        int(subset_config["validation_images"]),
        int(subset_config["minimum_validation_images_per_class"]),
        int(subset_config["seed"]) + 1,
    )

    train_list_path = output_directory / "train_images.txt"
    validation_list_path = output_directory / "validation_images.txt"

    write_image_list(train_list_path, selected_train)
    write_image_list(validation_list_path, selected_validation)

    pilot_yaml = {
        "path": str(dataset_root),
        "train": str(train_list_path.resolve()),
        "val": str(validation_list_path.resolve()),
        "test": str(layout.test_images),
        "nc": len(class_names),
        "names": class_names,
    }

    pilot_yaml_path = output_directory / "data.yaml"

    pilot_yaml_path.write_text(
        yaml.safe_dump(
            pilot_yaml,
            sort_keys=False,
            allow_unicode=True,
        ),
        encoding="utf-8",
    )

    subset_manifest = {
        "schema_version": 1,
        "created_at": datetime.now(UTC).isoformat(),
        "seed": int(subset_config["seed"]),
        "dataset_root": str(dataset_root),
        "source_data_yaml": str(layout.data_yaml),
        "source_data_yaml_sha256": sha256_file(layout.data_yaml),
        "class_names": class_names,
        "train": {
            "selected_images": len(selected_train),
            "class_image_counts": class_image_counts(selected_train),
            "image_list": str(train_list_path),
        },
        "validation": {
            "selected_images": len(selected_validation),
            "class_image_counts": class_image_counts(selected_validation),
            "image_list": str(validation_list_path),
        },
        "test_split_used": False,
    }

    write_json(output_directory / "subset_manifest.json", subset_manifest)

    validation_images = [record["image_path"] for record in selected_validation]

    return pilot_yaml_path, subset_manifest, validation_images


def download_generic_weights(
    weight_name: str,
    weights_directory: Path,
) -> Path:
    if weight_name not in ALLOWED_GENERIC_WEIGHTS:
        raise ValueError(f"Weight name is not an approved generic checkpoint: {weight_name}")

    if Path(weight_name).name != weight_name:
        raise ValueError("Generic weight names must not contain a path.")

    weights_directory.mkdir(parents=True, exist_ok=True)
    expected_path = weights_directory / weight_name

    if expected_path.is_file():
        return expected_path.resolve()

    original_directory = Path.cwd()

    try:
        os.chdir(weights_directory)
        model = YOLO(weight_name)

        checkpoint_path_value = getattr(model, "ckpt_path", None)

        del model
        gc.collect()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    finally:
        os.chdir(original_directory)

    if expected_path.is_file():
        return expected_path.resolve()

    if checkpoint_path_value:
        checkpoint_path = Path(checkpoint_path_value).resolve()

        if checkpoint_path.is_file():
            return checkpoint_path

    matching_paths = list(weights_directory.rglob(weight_name))

    if len(matching_paths) == 1:
        return matching_paths[0].resolve()

    raise FileNotFoundError(
        f"Official generic weight file was not found after loading: {weight_name}"
    )


def model_file_details(path: Path) -> dict[str, Any]:
    return {
        "path": str(path),
        "size_bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def is_cuda_oom(error: BaseException) -> bool:
    message = str(error).lower()

    return (
        "cuda out of memory" in message
        or "outofmemoryerror" in message
        or "cublas_status_alloc_failed" in message
    )


def possible_batch_sizes(initial_batch_size: int) -> list[int]:
    values: list[int] = []
    current = max(1, initial_batch_size)

    while current >= 1:
        if current not in values:
            values.append(current)

        if current == 1:
            break

        current = max(1, current // 2)

    return values


def extract_training_metrics(results: Any) -> dict[str, Any]:
    raw_metrics = getattr(results, "results_dict", {})

    if not isinstance(raw_metrics, dict):
        return {}

    return {str(key): scalar_value(value) for key, value in raw_metrics.items()}


def select_metric(
    metrics: dict[str, Any],
    possible_keys: tuple[str, ...],
) -> float | None:
    for key in possible_keys:
        value = metrics.get(key)

        if isinstance(value, int | float):
            return float(value)

    return None


def benchmark_model(
    model_path: Path,
    validation_images: list[Path],
    image_size: int,
    device: int,
) -> dict[str, Any]:
    if not validation_images:
        raise RuntimeError("No validation images are available for benchmarking.")

    benchmark_images = validation_images[:20]
    model = YOLO(str(model_path))

    warmup_images = benchmark_images[:2]

    for image_path in warmup_images:
        model.predict(
            source=str(image_path),
            imgsz=image_size,
            device=device,
            half=True,
            verbose=False,
        )

    torch.cuda.synchronize()

    latencies_ms: list[float] = []

    for image_path in benchmark_images:
        torch.cuda.synchronize()
        start_time = time.perf_counter()

        model.predict(
            source=str(image_path),
            imgsz=image_size,
            device=device,
            half=True,
            verbose=False,
        )

        torch.cuda.synchronize()

        latencies_ms.append((time.perf_counter() - start_time) * 1000.0)

    del model
    gc.collect()
    torch.cuda.empty_cache()

    return {
        "sample_count": len(latencies_ms),
        "mean_ms": statistics.fmean(latencies_ms),
        "p50_ms": percentile(latencies_ms, 0.50),
        "p95_ms": percentile(latencies_ms, 0.95),
        "minimum_ms": min(latencies_ms),
        "maximum_ms": max(latencies_ms),
        "measurements_ms": latencies_ms,
    }


def run_candidate(
    candidate: dict[str, Any],
    generic_weight_path: Path,
    pilot_data_yaml: Path,
    validation_images: list[Path],
    run_root: Path,
    config: dict[str, Any],
    run_timestamp: str,
) -> dict[str, Any]:
    candidate_name = str(candidate["name"])
    initial_batch_size = int(candidate["initial_batch_size"])
    training_config = config["training"]

    attempts: list[dict[str, Any]] = []

    for batch_size in possible_batch_sizes(initial_batch_size):
        attempt_name = f"{candidate_name}_{run_timestamp}_epoch1_batch{batch_size}"

        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        start_time = time.perf_counter()

        try:
            model = YOLO(str(generic_weight_path))

            results = model.train(
                data=str(pilot_data_yaml),
                epochs=int(training_config["epochs"]),
                imgsz=int(training_config["image_size"]),
                batch=batch_size,
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
                pretrained=True,
                resume=False,
                project=str(run_root),
                name=attempt_name,
                exist_ok=False,
                verbose=True,
            )

            duration_seconds = time.perf_counter() - start_time
            peak_allocated_gib = torch.cuda.max_memory_allocated() / (1024**3)
            peak_reserved_gib = torch.cuda.max_memory_reserved() / (1024**3)

            save_directory = Path(results.save_dir).resolve()
            best_model_path = save_directory / "weights" / "best.pt"
            last_model_path = save_directory / "weights" / "last.pt"

            if not best_model_path.is_file():
                raise FileNotFoundError(f"Pilot best model was not created: {best_model_path}")

            if not last_model_path.is_file():
                raise FileNotFoundError(f"Pilot last model was not created: {last_model_path}")

            metrics = extract_training_metrics(results)

            del model
            gc.collect()
            torch.cuda.empty_cache()

            latency = benchmark_model(
                best_model_path,
                validation_images,
                int(training_config["image_size"]),
                int(training_config["device"]),
            )

            result = {
                "candidate": candidate_name,
                "status": "PASS",
                "initialization_type": "official_generic_pretrained",
                "generic_weight": model_file_details(generic_weight_path),
                "resume": False,
                "batch_size": batch_size,
                "duration_seconds": duration_seconds,
                "peak_allocated_gib": peak_allocated_gib,
                "peak_reserved_gib": peak_reserved_gib,
                "metrics": metrics,
                "metrics_precision": select_metric(
                    metrics,
                    (
                        "metrics/precision(B)",
                        "metrics/precision",
                    ),
                ),
                "metrics_recall": select_metric(
                    metrics,
                    (
                        "metrics/recall(B)",
                        "metrics/recall",
                    ),
                ),
                "metrics_map50": select_metric(
                    metrics,
                    (
                        "metrics/mAP50(B)",
                        "metrics/mAP50",
                    ),
                ),
                "metrics_map50_95": select_metric(
                    metrics,
                    (
                        "metrics/mAP50-95(B)",
                        "metrics/mAP50-95",
                    ),
                ),
                "latency": latency,
                "latency_mean_ms": latency["mean_ms"],
                "latency_p50_ms": latency["p50_ms"],
                "latency_p95_ms": latency["p95_ms"],
                "best_model": str(best_model_path),
                "best_model_sha256": sha256_file(best_model_path),
                "last_model": str(last_model_path),
                "last_model_sha256": sha256_file(last_model_path),
                "run_directory": str(save_directory),
                "attempts": attempts,
                "error": None,
            }

            write_json(save_directory / "pilot_result.json", result)

            return result

        except Exception as error:
            duration_seconds = time.perf_counter() - start_time

            attempt = {
                "batch_size": batch_size,
                "duration_seconds": duration_seconds,
                "error_type": type(error).__name__,
                "error": str(error),
                "cuda_oom": is_cuda_oom(error),
            }

            attempts.append(attempt)

            gc.collect()

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            if not is_cuda_oom(error):
                return {
                    "candidate": candidate_name,
                    "status": "FAIL",
                    "initialization_type": "official_generic_pretrained",
                    "generic_weight": model_file_details(generic_weight_path),
                    "resume": False,
                    "batch_size": batch_size,
                    "duration_seconds": duration_seconds,
                    "peak_allocated_gib": None,
                    "peak_reserved_gib": None,
                    "metrics": {},
                    "metrics_precision": None,
                    "metrics_recall": None,
                    "metrics_map50": None,
                    "metrics_map50_95": None,
                    "latency": {},
                    "latency_mean_ms": None,
                    "latency_p50_ms": None,
                    "latency_p95_ms": None,
                    "best_model": None,
                    "best_model_sha256": None,
                    "last_model": None,
                    "last_model_sha256": None,
                    "run_directory": None,
                    "attempts": attempts,
                    "error": repr(error),
                }

    return {
        "candidate": candidate_name,
        "status": "FAIL",
        "initialization_type": "official_generic_pretrained",
        "generic_weight": model_file_details(generic_weight_path),
        "resume": False,
        "batch_size": None,
        "duration_seconds": None,
        "peak_allocated_gib": None,
        "peak_reserved_gib": None,
        "metrics": {},
        "metrics_precision": None,
        "metrics_recall": None,
        "metrics_map50": None,
        "metrics_map50_95": None,
        "latency": {},
        "latency_mean_ms": None,
        "latency_p50_ms": None,
        "latency_p95_ms": None,
        "best_model": None,
        "best_model_sha256": None,
        "last_model": None,
        "last_model_sha256": None,
        "run_directory": None,
        "attempts": attempts,
        "error": "All batch-size attempts failed with CUDA out-of-memory errors.",
    }


def validate_config(config: dict[str, Any]) -> None:
    policy = config.get("training_policy", {})

    if policy.get("resume") is not False:
        raise ValueError("Pilot training must set resume to false.")

    if policy.get("allow_historical_custom_ppe_checkpoint") is not False:
        raise ValueError("Historical custom PPE checkpoints must be disabled.")

    if policy.get("allow_official_generic_pretrained_weights") is not True:
        raise ValueError("Official generic pretrained weights must be allowed.")

    if policy.get("test_split_allowed") is not False:
        raise ValueError("The test split must not be used during pilot selection.")

    candidate_names: set[str] = set()

    for candidate in config.get("candidates", []):
        name = str(candidate.get("name"))
        weights = str(candidate.get("weights"))

        if name in candidate_names:
            raise ValueError(f"Duplicate candidate name: {name}")

        candidate_names.add(name)

        if weights not in ALLOWED_GENERIC_WEIGHTS:
            raise ValueError(f"Candidate uses an unapproved weight file: {weights}")

        if Path(weights).name != weights:
            raise ValueError(f"Candidate weight must not contain a path: {weights}")

    if candidate_names != {"yolo26n", "yolo26s"}:
        raise ValueError("The initial pilot must contain yolo26n and yolo26s.")


def main() -> int:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        type=Path,
        default=(PROJECT_ROOT / "configs" / "perception" / "pilot_training.json"),
    )

    arguments = parser.parse_args()

    config = load_json(arguments.config.resolve())
    validate_config(config)

    integration_report = load_json(
        PROJECT_ROOT / "data" / "manifests" / "ppe_dataset_integration_validation.json"
    )

    environment_report = load_json(
        PROJECT_ROOT / "reports" / "perception" / "training_environment.json"
    )

    if integration_report.get("status") != "PASS":
        raise RuntimeError("Dataset integration validation is not PASS.")

    if environment_report.get("training_ready") is not True:
        raise RuntimeError("Perception training environment is not ready.")

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available.")

    dataset_root = resolve_ppe_dataset_root(project_root=PROJECT_ROOT)

    timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")

    pilot_dataset_directory = (
        PROJECT_ROOT / "artifacts" / "perception" / "pilot_datasets" / timestamp
    )

    run_root = PROJECT_ROOT / "artifacts" / "runs" / "perception_pilots"

    weights_directory = PROJECT_ROOT / "artifacts" / "models" / "perception" / "generic_pretrained"

    reports_directory = PROJECT_ROOT / "reports" / "perception" / "pilot_runs"

    pilot_data_yaml, subset_manifest, validation_images = create_pilot_dataset(
        dataset_root,
        config,
        pilot_dataset_directory,
    )

    results: list[dict[str, Any]] = []

    for candidate in config["candidates"]:
        candidate_name = str(candidate["name"])
        weight_name = str(candidate["weights"])

        print()
        print("=" * 72)
        print(f"STARTING PILOT: {candidate_name}")
        print("=" * 72)

        generic_weight_path = download_generic_weights(
            weight_name,
            weights_directory,
        )

        print(f"Generic weight: {generic_weight_path}")
        print(f"Generic weight SHA256: {sha256_file(generic_weight_path)}")

        result = run_candidate(
            candidate,
            generic_weight_path,
            pilot_data_yaml,
            validation_images,
            run_root,
            config,
            timestamp,
        )

        results.append(result)

        print()
        print(f"Pilot status: {result['status']}")
        print(f"Candidate: {candidate_name}")
        print(f"Batch size: {result.get('batch_size')}")
        print(f"mAP50: {result.get('metrics_map50')}")
        print(f"mAP50-95: {result.get('metrics_map50_95')}")
        print(f"Latency mean ms: {result.get('latency_mean_ms')}")
        print(f"Error: {result.get('error')}")

    passed_results = [result for result in results if result["status"] == "PASS"]

    summary = {
        "schema_version": 1,
        "created_at": datetime.now(UTC).isoformat(),
        "experiment_type": config["experiment_type"],
        "status": "PASS" if passed_results else "FAIL",
        "purpose": (
            "One-epoch hardware and pipeline viability pilot. "
            "Metrics are not final model-selection results."
        ),
        "project_root": str(PROJECT_ROOT),
        "dataset_root": str(dataset_root),
        "pilot_dataset_yaml": str(pilot_data_yaml),
        "subset_manifest": subset_manifest,
        "training_policy": config["training_policy"],
        "training_config": config["training"],
        "environment": {
            "python": sys.version,
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "cuda_available": torch.cuda.is_available(),
            "gpu_name": torch.cuda.get_device_name(0),
            "gpu_total_memory_gib": round(
                torch.cuda.get_device_properties(0).total_memory / (1024**3),
                3,
            ),
            "ultralytics": package_version("ultralytics"),
        },
        "candidate_results": results,
        "passed_candidate_count": len(passed_results),
        "failed_candidate_count": len(results) - len(passed_results),
        "test_split_used": False,
        "commit_created": False,
        "push_performed": False,
    }

    timestamped_summary_path = reports_directory / f"pilot_summary_{timestamp}.json"

    latest_summary_path = reports_directory / "latest_pilot_summary.json"

    summary_csv_path = reports_directory / f"pilot_summary_{timestamp}.csv"

    write_json(timestamped_summary_path, summary)
    write_json(latest_summary_path, summary)
    write_csv(summary_csv_path, results)

    print()
    print("=" * 72)
    print("PPE DETECTOR PILOT SUMMARY")
    print("=" * 72)
    print(f"Status: {summary['status']}")
    print(f"Passed candidates: {len(passed_results)}")
    print(f"Failed candidates: {len(results) - len(passed_results)}")
    print(f"Summary: {timestamped_summary_path}")
    print(f"Latest summary: {latest_summary_path}")
    print(f"CSV: {summary_csv_path}")
    print("=" * 72)

    return 0 if passed_results else 1


if __name__ == "__main__":
    raise SystemExit(main())
