from __future__ import annotations

import csv
import hashlib
import json
import random
import statistics
import time
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch
import yaml
from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATASET_ROOT = PROJECT_ROOT / "Personal Protective Equipment - Combined Model.v1i.yolov8"

SOURCE_MODEL = (
    PROJECT_ROOT
    / "artifacts"
    / "runs"
    / "perception_production_optimized_100e"
    / "yolo26s_gpu90_fixed640_100e_seed42_20260723_003324"
    / "weights"
    / "best.pt"
)

EXPECTED_SOURCE_SHA256 = "23f916b8b7c99471955e4a64d820568c8b746bc0dbccd9dd0636122185aa0123"

CLASS_NAMES = [
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

WEAK_CLASS_REPEAT_FACTORS = {
    5: 2,
    8: 2,
    9: 2,
    10: 6,
    13: 3,
}

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp",
    ".tif",
    ".tiff",
}

ARTIFACT_ROOT = PROJECT_ROOT / "artifacts" / "perception" / "weak_class_fine_tuning"

RUNS_ROOT = PROJECT_ROOT / "artifacts" / "runs" / "perception_weak_class_fine_tuning"

REPORTS_ROOT = PROJECT_ROOT / "reports" / "perception" / "weak_class_fine_tuning"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as file_handle:
        while chunk := file_handle.read(1024 * 1024):
            digest.update(chunk)

    return digest.hexdigest()


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def image_to_label_path(image_path: Path) -> Path:
    relative_path = image_path.relative_to(DATASET_ROOT / "train" / "images")

    return (DATASET_ROOT / "train" / "labels" / relative_path).with_suffix(".txt")


def parse_label_classes(label_path: Path) -> list[int]:
    if not label_path.is_file():
        raise FileNotFoundError(f"Label was not found: {label_path}")

    class_ids: list[int] = []

    for line_number, raw_line in enumerate(
        label_path.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        stripped = raw_line.strip()

        if not stripped:
            continue

        parts = stripped.split()

        if len(parts) != 5:
            raise ValueError(f"Invalid YOLO label at {label_path}:{line_number}")

        class_id = int(float(parts[0]))

        if not 0 <= class_id < len(CLASS_NAMES):
            raise ValueError(f"Invalid class ID {class_id} at {label_path}:{line_number}")

        class_ids.append(class_id)

    return class_ids


def collect_training_images() -> list[Path]:
    train_images = DATASET_ROOT / "train" / "images"

    image_paths = sorted(
        path.resolve()
        for path in train_images.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )

    if not image_paths:
        raise RuntimeError("No training images were found.")

    return image_paths


def build_balanced_training_list(
    output_path: Path,
) -> dict[str, Any]:
    random_generator = random.Random(42)
    original_images = collect_training_images()

    selected_paths: list[Path] = []
    original_instance_counts: Counter[int] = Counter()
    repeated_image_counts: Counter[int] = Counter()
    weak_class_image_counts: Counter[int] = Counter()

    for image_path in original_images:
        label_path = image_to_label_path(image_path)
        class_ids = parse_label_classes(label_path)
        unique_class_ids = set(class_ids)

        original_instance_counts.update(class_ids)

        repeat_factor = 1

        for class_id, configured_factor in WEAK_CLASS_REPEAT_FACTORS.items():
            if class_id in unique_class_ids:
                weak_class_image_counts[class_id] += 1
                repeat_factor = max(
                    repeat_factor,
                    configured_factor,
                )

        selected_paths.extend([image_path] * repeat_factor)

        if repeat_factor > 1:
            for class_id in unique_class_ids:
                repeated_image_counts[class_id] += repeat_factor - 1

    random_generator.shuffle(selected_paths)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    output_path.write_text(
        "\n".join(str(path) for path in selected_paths) + "\n",
        encoding="utf-8",
    )

    return {
        "original_image_count": len(original_images),
        "balanced_entry_count": len(selected_paths),
        "expansion_ratio": len(selected_paths) / len(original_images),
        "weak_class_repeat_factors": {
            CLASS_NAMES[class_id]: repeat_factor
            for class_id, repeat_factor in (WEAK_CLASS_REPEAT_FACTORS.items())
        },
        "weak_class_image_counts": {
            CLASS_NAMES[class_id]: count
            for class_id, count in sorted(weak_class_image_counts.items())
        },
        "original_instance_counts": {
            CLASS_NAMES[class_id]: count
            for class_id, count in sorted(original_instance_counts.items())
        },
        "additional_repeated_image_entries_by_class": {
            CLASS_NAMES[class_id]: count
            for class_id, count in sorted(repeated_image_counts.items())
        },
    }


def create_runtime_yaml(
    train_list: Path,
    output_path: Path,
) -> None:
    payload = {
        "path": str(DATASET_ROOT),
        "train": str(train_list),
        "val": str(DATASET_ROOT / "valid" / "images"),
        "nc": len(CLASS_NAMES),
        "names": CLASS_NAMES,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)

    output_path.write_text(
        yaml.safe_dump(
            payload,
            sort_keys=False,
            allow_unicode=True,
        ),
        encoding="utf-8",
    )


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
        for class_name in CLASS_NAMES
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
        if not 0 <= class_id < len(CLASS_NAMES):
            continue

        class_name = CLASS_NAMES[class_id]

        ap_row = to_list(ap_values[position]) if position < len(ap_values) else []

        output[class_name] = {
            "precision": (
                float(precision_values[position]) if position < len(precision_values) else None
            ),
            "recall": (float(recall_values[position]) if position < len(recall_values) else None),
            "map50": (float(map50_values[position]) if position < len(map50_values) else None),
            "map50_95": (statistics.fmean(float(value) for value in ap_row) if ap_row else None),
        }

    return output


def overall_metrics(
    validation_results: Any,
) -> dict[str, float | None]:
    results_dict = getattr(
        validation_results,
        "results_dict",
        {},
    )

    if not isinstance(results_dict, dict):
        results_dict = {}

    def read_metric(*names: str) -> float | None:
        for name in names:
            value = results_dict.get(name)

            if value is None:
                continue

            if hasattr(value, "item"):
                value = value.item()

            try:
                return float(value)
            except (TypeError, ValueError):
                continue

        return None

    return {
        "precision": read_metric(
            "metrics/precision(B)",
            "metrics/precision",
        ),
        "recall": read_metric(
            "metrics/recall(B)",
            "metrics/recall",
        ),
        "map50": read_metric(
            "metrics/mAP50(B)",
            "metrics/mAP50",
        ),
        "map50_95": read_metric(
            "metrics/mAP50-95(B)",
            "metrics/mAP50-95",
        ),
    }


def validate_model(
    model_path: Path,
    data_yaml: Path,
    name: str,
    project: Path,
) -> dict[str, Any]:
    model = YOLO(str(model_path))

    results = model.val(
        data=str(data_yaml),
        split="val",
        imgsz=640,
        batch=16,
        device=0,
        workers=8,
        plots=True,
        save_json=False,
        project=str(project),
        name=name,
        exist_ok=True,
        verbose=True,
    )

    return {
        "overall": overall_metrics(results),
        "per_class": extract_per_class_metrics(results),
    }


def acceptance_decision(
    baseline: dict[str, Any],
    candidate: dict[str, Any],
) -> dict[str, Any]:
    baseline_weak = baseline["per_class"]["NO-Safety Vest"]
    candidate_weak = candidate["per_class"]["NO-Safety Vest"]

    baseline_fall = baseline["per_class"]["Fall-Detected"]
    candidate_fall = candidate["per_class"]["Fall-Detected"]

    overall_drop = float(baseline["overall"]["map50_95"]) - float(candidate["overall"]["map50_95"])

    fall_recall_drop = float(baseline_fall["recall"]) - float(candidate_fall["recall"])

    weak_recall_gain = float(candidate_weak["recall"]) - float(baseline_weak["recall"])

    weak_map50_gain = float(candidate_weak["map50"]) - float(baseline_weak["map50"])

    checks = {
        "no_safety_vest_recall_improved": (weak_recall_gain > 0.0),
        "no_safety_vest_map50_improved": (weak_map50_gain > 0.0),
        "overall_map50_95_drop_within_limit": (overall_drop <= 0.015),
        "fall_detected_recall_drop_within_limit": (fall_recall_drop <= 0.02),
    }

    return {
        "accepted": all(checks.values()),
        "checks": checks,
        "deltas": {
            "no_safety_vest_recall": weak_recall_gain,
            "no_safety_vest_map50": weak_map50_gain,
            "overall_map50_95": (
                float(candidate["overall"]["map50_95"]) - float(baseline["overall"]["map50_95"])
            ),
            "fall_detected_recall": (
                float(candidate_fall["recall"]) - float(baseline_fall["recall"])
            ),
        },
    }


def main() -> int:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available.")

    if not SOURCE_MODEL.is_file():
        raise FileNotFoundError(f"Source model was not found: {SOURCE_MODEL}")

    source_hash = sha256_file(SOURCE_MODEL)

    if source_hash != EXPECTED_SOURCE_SHA256:
        raise RuntimeError("Source model SHA256 validation failed.")

    timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")

    runtime_directory = ARTIFACT_ROOT / timestamp

    train_list = runtime_directory / "balanced_train.txt"

    runtime_yaml = runtime_directory / "data.yaml"

    balancing_report = build_balanced_training_list(train_list)

    create_runtime_yaml(
        train_list,
        runtime_yaml,
    )

    write_json(
        runtime_directory / "balancing_report.json",
        balancing_report,
    )

    print()
    print("=" * 72)
    print("WEAK-CLASS BALANCED FINE-TUNING")
    print("=" * 72)
    print(f"Source model: {SOURCE_MODEL}")
    print(f"Source SHA256: {source_hash}")
    print(f"Original images: {balancing_report['original_image_count']}")
    print(f"Balanced entries: {balancing_report['balanced_entry_count']}")
    print(f"Expansion ratio: {balancing_report['expansion_ratio']:.3f}")
    print("Epochs: 20")
    print("Image size: 640")
    print("Batch size: 16")
    print("Test split used: False")
    print("=" * 72)

    baseline_validation = validate_model(
        SOURCE_MODEL,
        runtime_yaml,
        "baseline_validation",
        runtime_directory,
    )

    run_name = f"yolo26s_weak_class_20e_{timestamp}"

    model = YOLO(str(SOURCE_MODEL))
    started_at = time.perf_counter()

    training_results = model.train(
        data=str(runtime_yaml),
        epochs=20,
        imgsz=640,
        batch=16,
        device=0,
        workers=8,
        amp=True,
        cache="disk",
        optimizer="AdamW",
        lr0=0.0005,
        lrf=0.05,
        weight_decay=0.0005,
        warmup_epochs=1.0,
        patience=20,
        cos_lr=True,
        multi_scale=0.0,
        mosaic=0.5,
        mixup=0.05,
        close_mosaic=5,
        seed=43,
        deterministic=False,
        fraction=1.0,
        pretrained=True,
        resume=False,
        save=True,
        save_period=5,
        plots=True,
        val=True,
        project=str(RUNS_ROOT),
        name=run_name,
        exist_ok=False,
        verbose=True,
    )

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
            raise FileNotFoundError(f"Required artifact was not created: {required_path}")

    candidate_validation = validate_model(
        best_model,
        runtime_yaml,
        "candidate_validation",
        runtime_directory,
    )

    decision = acceptance_decision(
        baseline_validation,
        candidate_validation,
    )

    with results_csv.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as file_handle:
        completed_epochs = sum(1 for _ in csv.DictReader(file_handle))

    summary = {
        "schema_version": 1,
        "created_at": datetime.now(UTC).isoformat(),
        "status": "PASS",
        "source_model": str(SOURCE_MODEL),
        "source_model_sha256": source_hash,
        "runtime_data_yaml": str(runtime_yaml),
        "balancing_report": balancing_report,
        "epochs_requested": 20,
        "epochs_completed": completed_epochs,
        "duration_seconds": duration_seconds,
        "test_split_used": False,
        "baseline_validation": baseline_validation,
        "candidate_validation": candidate_validation,
        "acceptance_decision": decision,
        "candidate_best_model": str(best_model),
        "candidate_best_model_sha256": sha256_file(best_model),
        "candidate_last_model": str(last_model),
        "candidate_last_model_sha256": sha256_file(last_model),
        "run_directory": str(run_directory),
    }

    timestamped_report = REPORTS_ROOT / f"summary_{timestamp}.json"

    latest_report = REPORTS_ROOT / "latest_summary.json"

    write_json(timestamped_report, summary)
    write_json(latest_report, summary)

    print()
    print("=" * 72)
    print("WEAK-CLASS FINE-TUNING SUMMARY")
    print("=" * 72)
    print(f"Epochs completed: {completed_epochs}")
    print(f"Baseline overall mAP50-95: {baseline_validation['overall']['map50_95']}")
    print(f"Candidate overall mAP50-95: {candidate_validation['overall']['map50_95']}")
    print(
        "Baseline NO-Safety Vest recall: "
        f"{baseline_validation['per_class']['NO-Safety Vest']['recall']}"
    )
    print(
        "Candidate NO-Safety Vest recall: "
        f"{candidate_validation['per_class']['NO-Safety Vest']['recall']}"
    )
    print(
        "Baseline NO-Safety Vest mAP50: "
        f"{baseline_validation['per_class']['NO-Safety Vest']['map50']}"
    )
    print(
        "Candidate NO-Safety Vest mAP50: "
        f"{candidate_validation['per_class']['NO-Safety Vest']['map50']}"
    )
    print(f"Candidate accepted: {decision['accepted']}")
    print(f"Candidate model: {best_model}")
    print(f"Report: {latest_report}")
    print("=" * 72)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
