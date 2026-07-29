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

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp",
    ".tif",
    ".tiff",
}

TRAIN_ENTRY_LIMIT = 8500
VALIDATION_IMAGE_LIMIT = 1000
MAX_EPOCHS = 100
EARLY_STOPPING_PATIENCE = 12

TARGET_CLASS_WEIGHTS = {
    0: 3.0,
    5: 3.0,
    8: 3.5,
    9: 3.0,
    10: 14.0,
    11: 1.5,
    13: 5.0,
}

ARTIFACT_ROOT = PROJECT_ROOT / "artifacts" / "perception" / "targeted_fast_100e"

RUNS_ROOT = PROJECT_ROOT / "artifacts" / "runs" / "perception_targeted_fast_100e"

REPORTS_ROOT = PROJECT_ROOT / "reports" / "perception" / "targeted_fast_100e"


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


def list_images(split: str) -> list[Path]:
    image_directory = DATASET_ROOT / split / "images"

    images = sorted(
        path.resolve()
        for path in image_directory.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )

    if not images:
        raise RuntimeError(f"No images were found in split: {split}")

    return images


def image_to_label(image_path: Path, split: str) -> Path:
    relative_path = image_path.relative_to(DATASET_ROOT / split / "images")

    return (DATASET_ROOT / split / "labels" / relative_path).with_suffix(".txt")


def parse_class_ids(label_path: Path) -> list[int]:
    if not label_path.is_file():
        raise FileNotFoundError(f"Label file was not found: {label_path}")

    class_ids: list[int] = []

    for line_number, raw_line in enumerate(
        label_path.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        stripped = raw_line.strip()

        if not stripped:
            continue

        fields = stripped.split()

        if len(fields) != 5:
            raise ValueError(f"Invalid label at {label_path}:{line_number}")

        class_id = int(float(fields[0]))

        if not 0 <= class_id < len(CLASS_NAMES):
            raise ValueError(f"Invalid class ID at {label_path}:{line_number}")

        class_ids.append(class_id)

    return class_ids


def image_weight(class_ids: list[int]) -> float:
    unique_ids = set(class_ids)
    weight = 1.0

    for class_id in unique_ids:
        weight += TARGET_CLASS_WEIGHTS.get(
            class_id,
            0.0,
        )

    has_person = 11 in unique_ids
    has_vest_label = 10 in unique_ids or 13 in unique_ids

    if has_person and not has_vest_label:
        weight += 3.0

    return weight


def build_targeted_training_list(
    output_path: Path,
) -> dict[str, Any]:
    random_generator = random.Random(43)
    images = list_images("train")

    weights: list[float] = []
    classes_by_image: dict[Path, list[int]] = {}

    mandatory_images: list[Path] = []

    for image_path in images:
        class_ids = parse_class_ids(
            image_to_label(
                image_path,
                "train",
            )
        )

        classes_by_image[image_path] = class_ids
        weights.append(image_weight(class_ids))

        if 10 in set(class_ids):
            mandatory_images.append(image_path)

    selected_entries = list(mandatory_images)

    remaining_count = TRAIN_ENTRY_LIMIT - len(selected_entries)

    if remaining_count < 0:
        selected_entries = random_generator.sample(
            selected_entries,
            TRAIN_ENTRY_LIMIT,
        )
    else:
        selected_entries.extend(
            random_generator.choices(
                population=images,
                weights=weights,
                k=remaining_count,
            )
        )

    random_generator.shuffle(selected_entries)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path.write_text(
        "\n".join(str(path) for path in selected_entries) + "\n",
        encoding="utf-8",
    )

    selected_class_instances: Counter[int] = Counter()
    selected_class_images: Counter[int] = Counter()

    for image_path in selected_entries:
        class_ids = classes_by_image[image_path]

        selected_class_instances.update(class_ids)
        selected_class_images.update(set(class_ids))

    return {
        "source_image_count": len(images),
        "mandatory_no_safety_vest_image_count": len(mandatory_images),
        "selected_entry_count": len(selected_entries),
        "expected_batches_per_epoch": (len(selected_entries) + 15) // 16,
        "selected_class_image_entries": {
            CLASS_NAMES[class_id]: count
            for class_id, count in sorted(selected_class_images.items())
        },
        "selected_class_instances": {
            CLASS_NAMES[class_id]: count
            for class_id, count in sorted(selected_class_instances.items())
        },
    }


def build_validation_subset(
    output_path: Path,
) -> dict[str, Any]:
    random_generator = random.Random(44)
    images = list_images("valid")

    classes_by_image: dict[Path, set[int]] = {}
    images_by_class: dict[int, list[Path]] = {class_id: [] for class_id in range(len(CLASS_NAMES))}

    for image_path in images:
        unique_ids = set(
            parse_class_ids(
                image_to_label(
                    image_path,
                    "valid",
                )
            )
        )

        classes_by_image[image_path] = unique_ids

        for class_id in unique_ids:
            images_by_class[class_id].append(image_path)

    selected: list[Path] = []
    selected_set: set[Path] = set()

    def add_image(image_path: Path) -> None:
        if len(selected) < VALIDATION_IMAGE_LIMIT and image_path not in selected_set:
            selected.append(image_path)
            selected_set.add(image_path)

    for image_path in images_by_class[10]:
        add_image(image_path)

    priority_order = [
        0,
        13,
        8,
        5,
        9,
        11,
        12,
        3,
        1,
        2,
        4,
        6,
        7,
    ]

    for class_id in priority_order:
        candidates = list(images_by_class[class_id])

        random_generator.shuffle(candidates)

        for image_path in candidates[:120]:
            add_image(image_path)

    remaining_images = [image_path for image_path in images if image_path not in selected_set]

    random_generator.shuffle(remaining_images)

    for image_path in remaining_images:
        add_image(image_path)

        if len(selected) >= VALIDATION_IMAGE_LIMIT:
            break

    random_generator.shuffle(selected)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path.write_text(
        "\n".join(str(path) for path in selected) + "\n",
        encoding="utf-8",
    )

    class_image_counts: Counter[int] = Counter()

    for image_path in selected:
        class_image_counts.update(classes_by_image[image_path])

    missing_classes = [
        CLASS_NAMES[class_id]
        for class_id in range(len(CLASS_NAMES))
        if class_image_counts[class_id] == 0
    ]

    if missing_classes:
        raise RuntimeError("Validation subset is missing classes: " + ", ".join(missing_classes))

    return {
        "source_image_count": len(images),
        "selected_image_count": len(selected),
        "class_image_counts": {
            CLASS_NAMES[class_id]: count for class_id, count in sorted(class_image_counts.items())
        },
    }


def write_runtime_yaml(
    train_list: Path,
    validation_list: Path,
    output_path: Path,
) -> None:
    payload = {
        "path": str(DATASET_ROOT),
        "train": str(train_list),
        "val": str(validation_list),
        "nc": len(CLASS_NAMES),
        "names": CLASS_NAMES,
    }

    output_path.write_text(
        yaml.safe_dump(
            payload,
            sort_keys=False,
            allow_unicode=True,
        ),
        encoding="utf-8",
    )


def write_full_validation_yaml(
    output_path: Path,
) -> None:
    payload = {
        "path": str(DATASET_ROOT),
        "train": str(DATASET_ROOT / "train" / "images"),
        "val": str(DATASET_ROOT / "valid" / "images"),
        "nc": len(CLASS_NAMES),
        "names": CLASS_NAMES,
    }

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


def mean_ap_row(value: Any) -> float | None:
    row = to_list(value)

    numeric_values = [float(item) for item in row if isinstance(item, (int, float))]

    if not numeric_values:
        return None

    return statistics.fmean(numeric_values)


def extract_metrics(
    validation_results: Any,
) -> dict[str, Any]:
    results_dict = getattr(
        validation_results,
        "results_dict",
        {},
    )

    if not isinstance(results_dict, dict):
        results_dict = {}

    def read_metric(*keys: str) -> float | None:
        for key in keys:
            value = results_dict.get(key)

            if value is None:
                continue

            if hasattr(value, "item"):
                value = value.item()

            try:
                return float(value)
            except (TypeError, ValueError):
                continue

        return None

    per_class = {
        class_name: {
            "precision": None,
            "recall": None,
            "map50": None,
            "map50_95": None,
        }
        for class_name in CLASS_NAMES
    }

    box_metrics = getattr(
        validation_results,
        "box",
        None,
    )

    if box_metrics is not None:
        class_indices = [
            int(value)
            for value in to_list(
                getattr(
                    box_metrics,
                    "ap_class_index",
                    None,
                )
            )
        ]

        precision_values = to_list(getattr(box_metrics, "p", None))

        recall_values = to_list(getattr(box_metrics, "r", None))

        map50_values = to_list(getattr(box_metrics, "ap50", None))

        ap_values = to_list(getattr(box_metrics, "ap", None))

        for position, class_id in enumerate(class_indices):
            if not 0 <= class_id < len(CLASS_NAMES):
                continue

            per_class[CLASS_NAMES[class_id]] = {
                "precision": (
                    float(precision_values[position]) if position < len(precision_values) else None
                ),
                "recall": (
                    float(recall_values[position]) if position < len(recall_values) else None
                ),
                "map50": (float(map50_values[position]) if position < len(map50_values) else None),
                "map50_95": (
                    mean_ap_row(ap_values[position]) if position < len(ap_values) else None
                ),
            }

    return {
        "overall": {
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
        },
        "per_class": per_class,
    }


def validate_model(
    model_path: Path,
    data_yaml: Path,
    output_directory: Path,
    name: str,
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
        project=str(output_directory),
        name=name,
        exist_ok=True,
        verbose=True,
    )

    return extract_metrics(results)


def acceptance_decision(
    baseline: dict[str, Any],
    candidate: dict[str, Any],
) -> dict[str, Any]:
    baseline_weak = baseline["per_class"]["NO-Safety Vest"]

    candidate_weak = candidate["per_class"]["NO-Safety Vest"]

    baseline_fall = baseline["per_class"]["Fall-Detected"]

    candidate_fall = candidate["per_class"]["Fall-Detected"]

    weak_recall_gain = float(candidate_weak["recall"]) - float(baseline_weak["recall"])

    weak_map50_gain = float(candidate_weak["map50"]) - float(baseline_weak["map50"])

    overall_map_drop = float(baseline["overall"]["map50_95"]) - float(
        candidate["overall"]["map50_95"]
    )

    fall_recall_drop = float(baseline_fall["recall"]) - float(candidate_fall["recall"])

    checks = {
        "no_safety_vest_recall_gain_at_least_0_05": (weak_recall_gain >= 0.05),
        "no_safety_vest_map50_improved": (weak_map50_gain > 0.0),
        "overall_map50_95_drop_at_most_0_015": (overall_map_drop <= 0.015),
        "fall_recall_drop_at_most_0_02": (fall_recall_drop <= 0.02),
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


def count_completed_epochs(results_csv: Path) -> int:
    with results_csv.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as file_handle:
        return sum(1 for _ in csv.DictReader(file_handle))


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
    runtime_directory.mkdir(
        parents=True,
        exist_ok=False,
    )

    train_list = runtime_directory / "targeted_train.txt"

    validation_list = runtime_directory / "early_stopping_validation.txt"

    runtime_yaml = runtime_directory / "targeted_data.yaml"

    full_validation_yaml = runtime_directory / "full_validation_data.yaml"

    training_manifest = build_targeted_training_list(train_list)

    validation_manifest = build_validation_subset(validation_list)

    write_runtime_yaml(
        train_list,
        validation_list,
        runtime_yaml,
    )

    write_full_validation_yaml(full_validation_yaml)

    manifest = {
        "schema_version": 1,
        "created_at": datetime.now(UTC).isoformat(),
        "train": training_manifest,
        "validation": validation_manifest,
        "max_epochs": MAX_EPOCHS,
        "early_stopping_patience": (EARLY_STOPPING_PATIENCE),
        "image_size": 640,
        "batch_size": 16,
        "test_split_used": False,
    }

    write_json(
        runtime_directory / "manifest.json",
        manifest,
    )

    print()
    print("=" * 72)
    print("FAST TARGETED PPE FINE-TUNING")
    print("=" * 72)
    print(f"Source model: {SOURCE_MODEL}")
    print(f"Source SHA256: {source_hash}")
    print(f"Training entries: {training_manifest['selected_entry_count']}")
    print(f"Expected batches per epoch: {training_manifest['expected_batches_per_epoch']}")
    print(f"Early-stopping validation images: {validation_manifest['selected_image_count']}")
    print("Maximum epochs: 100")
    print("Early stopping patience: 12")
    print("Image size: 640")
    print("Batch size: 16")
    print("Frozen layers: 10")
    print("Test split used: False")
    print("=" * 72)

    run_name = f"yolo26s_targeted_fast100_{timestamp}"

    model = YOLO(str(SOURCE_MODEL))
    started_at = time.perf_counter()

    training_results = model.train(
        data=str(runtime_yaml),
        epochs=MAX_EPOCHS,
        patience=EARLY_STOPPING_PATIENCE,
        imgsz=640,
        batch=16,
        device=0,
        workers=8,
        cache="disk",
        amp=True,
        freeze=10,
        optimizer="AdamW",
        lr0=0.0003,
        lrf=0.05,
        weight_decay=0.0005,
        warmup_epochs=1.0,
        cos_lr=True,
        mosaic=0.5,
        mixup=0.0,
        close_mosaic=10,
        multi_scale=0.0,
        seed=43,
        deterministic=False,
        save=True,
        save_period=5,
        plots=True,
        val=True,
        pretrained=True,
        resume=False,
        project=str(RUNS_ROOT),
        name=run_name,
        exist_ok=False,
        verbose=True,
    )

    training_duration = time.perf_counter() - started_at

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

    completed_epochs = count_completed_epochs(results_csv)

    average_epoch_seconds = training_duration / completed_epochs

    print()
    print("Running full validation for the baseline...")

    baseline_validation = validate_model(
        SOURCE_MODEL,
        full_validation_yaml,
        runtime_directory,
        "baseline_full_validation",
    )

    print()
    print("Running full validation for the candidate...")

    candidate_validation = validate_model(
        best_model,
        full_validation_yaml,
        runtime_directory,
        "candidate_full_validation",
    )

    decision = acceptance_decision(
        baseline_validation,
        candidate_validation,
    )

    selected_model = best_model if decision["accepted"] else SOURCE_MODEL

    selection_reason = (
        "The targeted candidate passed all acceptance checks."
        if decision["accepted"]
        else (
            "The targeted candidate did not pass every acceptance "
            "check, so the original 100-epoch baseline remains selected."
        )
    )

    summary = {
        "schema_version": 1,
        "created_at": datetime.now(UTC).isoformat(),
        "status": "PASS",
        "maximum_epochs": MAX_EPOCHS,
        "epochs_completed": completed_epochs,
        "early_stopping_enabled": True,
        "early_stopping_patience": (EARLY_STOPPING_PATIENCE),
        "early_stopping_triggered": (completed_epochs < MAX_EPOCHS),
        "training_duration_seconds": (training_duration),
        "average_epoch_seconds": (average_epoch_seconds),
        "target_epoch_seconds": 180,
        "average_epoch_target_met": (average_epoch_seconds <= 180),
        "training_manifest": training_manifest,
        "validation_manifest": validation_manifest,
        "baseline_validation": baseline_validation,
        "candidate_validation": candidate_validation,
        "acceptance_decision": decision,
        "candidate_best_model": str(best_model),
        "candidate_best_model_sha256": (sha256_file(best_model)),
        "candidate_last_model": str(last_model),
        "candidate_last_model_sha256": (sha256_file(last_model)),
        "selected_model": str(selected_model),
        "selected_model_sha256": (sha256_file(selected_model)),
        "selection_reason": selection_reason,
        "test_split_used": False,
        "run_directory": str(run_directory),
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

    print()
    print("=" * 72)
    print("FAST TARGETED TRAINING SUMMARY")
    print("=" * 72)
    print("Status: PASS")
    print("Maximum epochs: 100")
    print(f"Epochs completed: {completed_epochs}")
    print(f"Early stopping triggered: {summary['early_stopping_triggered']}")
    print(f"Average epoch seconds: {average_epoch_seconds:.2f}")
    print(f"Average epoch target met: {summary['average_epoch_target_met']}")
    print(f"Candidate accepted: {decision['accepted']}")
    print(f"NO-Safety Vest recall delta: {decision['deltas']['no_safety_vest_recall']}")
    print(f"NO-Safety Vest mAP50 delta: {decision['deltas']['no_safety_vest_map50']}")
    print(f"Overall mAP50-95 delta: {decision['deltas']['overall_map50_95']}")
    print(f"Selected model: {selected_model}")
    print(f"Selected SHA256: {summary['selected_model_sha256']}")
    print(f"Report: {latest_summary}")
    print("=" * 72)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
