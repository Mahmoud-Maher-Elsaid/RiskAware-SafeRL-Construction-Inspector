from __future__ import annotations

import csv
import hashlib
import json
import math
import sys
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml
from PIL import Image, UnidentifiedImageError

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp",
    ".tif",
    ".tiff",
}

SPLIT_DIRECTORIES = {
    "train": "train",
    "valid": "valid",
    "test": "test",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as file_handle:
        for chunk in iter(lambda: file_handle.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


def normalize_names(raw_names: Any) -> list[str]:
    if isinstance(raw_names, list):
        return [str(name) for name in raw_names]

    if isinstance(raw_names, dict):
        normalized_items: list[tuple[int, str]] = []

        for key, value in raw_names.items():
            normalized_items.append((int(key), str(value)))

        normalized_items.sort(key=lambda item: item[0])

        expected_ids = list(range(len(normalized_items)))
        actual_ids = [item[0] for item in normalized_items]

        if actual_ids != expected_ids:
            raise RuntimeError(
                f"Dataset class IDs are not contiguous from zero. Found: {actual_ids}"
            )

        return [item[1] for item in normalized_items]

    raise RuntimeError("data.yaml must define names as a list or an integer-keyed mapping.")


def relative_key(path: Path, root: Path) -> str:
    relative = path.relative_to(root)
    return relative.with_suffix("").as_posix().lower()


def read_image_size(path: Path) -> tuple[int, int]:
    with Image.open(path) as image:
        width, height = image.size
        image.verify()

    return int(width), int(height)


def main() -> int:
    if len(sys.argv) != 2:
        raise RuntimeError("Usage: audit_ppe_dataset.py <dataset-root>")

    dataset_root = Path(sys.argv[1]).resolve()
    project_root = Path(__file__).resolve().parents[1]
    data_yaml_path = dataset_root / "data.yaml"

    if not dataset_root.is_dir():
        raise RuntimeError(f"Dataset root was not found: {dataset_root}")

    if not data_yaml_path.is_file():
        raise RuntimeError(f"data.yaml was not found: {data_yaml_path}")

    with data_yaml_path.open("r", encoding="utf-8") as file_handle:
        data_yaml = yaml.safe_load(file_handle)

    if not isinstance(data_yaml, dict):
        raise RuntimeError("data.yaml does not contain a mapping.")

    class_names = normalize_names(data_yaml.get("names"))
    declared_class_count = data_yaml.get("nc")

    if declared_class_count is not None:
        declared_class_count = int(declared_class_count)

        if declared_class_count != len(class_names):
            raise RuntimeError(
                "data.yaml class count mismatch. "
                f"nc={declared_class_count}, names={len(class_names)}"
            )

    manifest_directory = project_root / "data" / "manifests"
    manifest_directory.mkdir(parents=True, exist_ok=True)

    split_reports: dict[str, dict[str, Any]] = {}
    class_instance_counts: Counter[int] = Counter()
    class_image_counts: Counter[int] = Counter()

    all_image_dimensions: Counter[str] = Counter()

    total_invalid_rows = 0
    total_invalid_class_ids = 0
    total_invalid_coordinates = 0
    total_out_of_bounds_boxes = 0
    total_zero_area_boxes = 0
    total_duplicate_boxes = 0
    total_corrupt_images = 0
    total_missing_labels = 0
    total_orphan_labels = 0
    total_background_images = 0
    total_instances = 0

    all_issues: list[dict[str, Any]] = []

    for split_name, directory_name in SPLIT_DIRECTORIES.items():
        images_directory = dataset_root / directory_name / "images"
        labels_directory = dataset_root / directory_name / "labels"

        if not images_directory.is_dir():
            raise RuntimeError(f"Images directory was not found: {images_directory}")

        if not labels_directory.is_dir():
            raise RuntimeError(f"Labels directory was not found: {labels_directory}")

        image_paths = sorted(
            path
            for path in images_directory.rglob("*")
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        )

        label_paths = sorted(labels_directory.rglob("*.txt"))

        image_keys = {relative_key(path, images_directory): path for path in image_paths}

        label_keys = {relative_key(path, labels_directory): path for path in label_paths}

        missing_label_keys = sorted(set(image_keys) - set(label_keys))

        orphan_label_keys = sorted(set(label_keys) - set(image_keys))

        split_invalid_rows = 0
        split_invalid_class_ids = 0
        split_invalid_coordinates = 0
        split_out_of_bounds_boxes = 0
        split_zero_area_boxes = 0
        split_duplicate_boxes = 0
        split_corrupt_images = 0
        split_background_images = 0
        split_instances = 0

        print(f"Auditing {split_name}: {len(image_paths)} images, {len(label_paths)} labels")

        for image_index, image_path in enumerate(image_paths, start=1):
            image_key = relative_key(
                image_path,
                images_directory,
            )

            label_path = label_keys.get(image_key)

            try:
                width, height = read_image_size(image_path)
                all_image_dimensions[f"{width}x{height}"] += 1
            except (
                OSError,
                ValueError,
                UnidentifiedImageError,
            ) as error:
                split_corrupt_images += 1
                all_issues.append(
                    {
                        "split": split_name,
                        "type": "corrupt_image",
                        "path": str(image_path),
                        "detail": repr(error),
                    }
                )
                continue

            if label_path is None:
                continue

            try:
                label_text = label_path.read_text(encoding="utf-8")
            except UnicodeDecodeError as error:
                split_invalid_rows += 1
                all_issues.append(
                    {
                        "split": split_name,
                        "type": "label_encoding_error",
                        "path": str(label_path),
                        "detail": repr(error),
                    }
                )
                continue

            label_lines = [line.strip() for line in label_text.splitlines() if line.strip()]

            if not label_lines:
                split_background_images += 1
                continue

            classes_in_image: set[int] = set()
            boxes_seen: set[tuple[int, float, float, float, float]] = set()

            for line_number, line in enumerate(
                label_lines,
                start=1,
            ):
                values = line.split()

                if len(values) != 5:
                    split_invalid_rows += 1
                    all_issues.append(
                        {
                            "split": split_name,
                            "type": "invalid_label_column_count",
                            "path": str(label_path),
                            "line": line_number,
                            "content": line,
                        }
                    )
                    continue

                try:
                    class_id_float = float(values[0])
                    class_id = int(class_id_float)
                    x_center = float(values[1])
                    y_center = float(values[2])
                    box_width = float(values[3])
                    box_height = float(values[4])
                except ValueError:
                    split_invalid_rows += 1
                    all_issues.append(
                        {
                            "split": split_name,
                            "type": "non_numeric_label",
                            "path": str(label_path),
                            "line": line_number,
                            "content": line,
                        }
                    )
                    continue

                if (
                    not math.isfinite(class_id_float)
                    or not math.isfinite(x_center)
                    or not math.isfinite(y_center)
                    or not math.isfinite(box_width)
                    or not math.isfinite(box_height)
                ):
                    split_invalid_rows += 1
                    all_issues.append(
                        {
                            "split": split_name,
                            "type": "non_finite_label",
                            "path": str(label_path),
                            "line": line_number,
                            "content": line,
                        }
                    )
                    continue

                if class_id_float != class_id:
                    split_invalid_class_ids += 1
                    all_issues.append(
                        {
                            "split": split_name,
                            "type": "non_integer_class_id",
                            "path": str(label_path),
                            "line": line_number,
                            "content": line,
                        }
                    )
                    continue

                if not 0 <= class_id < len(class_names):
                    split_invalid_class_ids += 1
                    all_issues.append(
                        {
                            "split": split_name,
                            "type": "class_id_out_of_range",
                            "path": str(label_path),
                            "line": line_number,
                            "class_id": class_id,
                        }
                    )
                    continue

                coordinate_values = (
                    x_center,
                    y_center,
                    box_width,
                    box_height,
                )

                if any(value < 0.0 or value > 1.0 for value in coordinate_values):
                    split_invalid_coordinates += 1
                    all_issues.append(
                        {
                            "split": split_name,
                            "type": "coordinate_outside_normalized_range",
                            "path": str(label_path),
                            "line": line_number,
                            "content": line,
                        }
                    )
                    continue

                if box_width <= 0.0 or box_height <= 0.0:
                    split_zero_area_boxes += 1
                    all_issues.append(
                        {
                            "split": split_name,
                            "type": "zero_area_box",
                            "path": str(label_path),
                            "line": line_number,
                            "content": line,
                        }
                    )
                    continue

                tolerance = 1e-6

                left = x_center - (box_width / 2.0)
                right = x_center + (box_width / 2.0)
                top = y_center - (box_height / 2.0)
                bottom = y_center + (box_height / 2.0)

                if (
                    left < -tolerance
                    or top < -tolerance
                    or right > 1.0 + tolerance
                    or bottom > 1.0 + tolerance
                ):
                    split_out_of_bounds_boxes += 1
                    all_issues.append(
                        {
                            "split": split_name,
                            "type": "box_extends_outside_image",
                            "path": str(label_path),
                            "line": line_number,
                            "content": line,
                        }
                    )

                normalized_box = (
                    class_id,
                    round(x_center, 8),
                    round(y_center, 8),
                    round(box_width, 8),
                    round(box_height, 8),
                )

                if normalized_box in boxes_seen:
                    split_duplicate_boxes += 1
                    all_issues.append(
                        {
                            "split": split_name,
                            "type": "duplicate_box",
                            "path": str(label_path),
                            "line": line_number,
                            "content": line,
                        }
                    )
                else:
                    boxes_seen.add(normalized_box)

                class_instance_counts[class_id] += 1
                classes_in_image.add(class_id)
                split_instances += 1

            for class_id in classes_in_image:
                class_image_counts[class_id] += 1

            if image_index % 2000 == 0:
                print(f"  Processed {image_index}/{len(image_paths)} images")

        split_report = {
            "images": len(image_paths),
            "labels": len(label_paths),
            "missing_labels": len(missing_label_keys),
            "orphan_labels": len(orphan_label_keys),
            "background_images": split_background_images,
            "instances": split_instances,
            "invalid_rows": split_invalid_rows,
            "invalid_class_ids": split_invalid_class_ids,
            "invalid_coordinates": split_invalid_coordinates,
            "out_of_bounds_boxes": split_out_of_bounds_boxes,
            "zero_area_boxes": split_zero_area_boxes,
            "duplicate_boxes": split_duplicate_boxes,
            "corrupt_images": split_corrupt_images,
        }

        split_reports[split_name] = split_report

        total_missing_labels += len(missing_label_keys)
        total_orphan_labels += len(orphan_label_keys)
        total_background_images += split_background_images
        total_instances += split_instances
        total_invalid_rows += split_invalid_rows
        total_invalid_class_ids += split_invalid_class_ids
        total_invalid_coordinates += split_invalid_coordinates
        total_out_of_bounds_boxes += split_out_of_bounds_boxes
        total_zero_area_boxes += split_zero_area_boxes
        total_duplicate_boxes += split_duplicate_boxes
        total_corrupt_images += split_corrupt_images

        for missing_key in missing_label_keys:
            all_issues.append(
                {
                    "split": split_name,
                    "type": "missing_label",
                    "key": missing_key,
                    "image_path": str(image_keys[missing_key]),
                }
            )

        for orphan_key in orphan_label_keys:
            all_issues.append(
                {
                    "split": split_name,
                    "type": "orphan_label",
                    "key": orphan_key,
                    "label_path": str(label_keys[orphan_key]),
                }
            )

    critical_issue_count = sum(
        (
            total_missing_labels,
            total_orphan_labels,
            total_invalid_rows,
            total_invalid_class_ids,
            total_invalid_coordinates,
            total_zero_area_boxes,
            total_corrupt_images,
        )
    )

    warning_issue_count = total_out_of_bounds_boxes + total_duplicate_boxes

    if critical_issue_count > 0:
        audit_status = "FAIL"
    elif warning_issue_count > 0:
        audit_status = "WARN"
    else:
        audit_status = "PASS"

    class_rows = []

    for class_id, class_name in enumerate(class_names):
        class_rows.append(
            {
                "class_id": class_id,
                "class_name": class_name,
                "instance_count": class_instance_counts[class_id],
                "image_count": class_image_counts[class_id],
            }
        )

    audit_report = {
        "schema_version": 1,
        "created_at": datetime.now(UTC).isoformat(),
        "status": audit_status,
        "project_root": str(project_root),
        "dataset_root": str(dataset_root),
        "data_yaml": str(data_yaml_path),
        "data_yaml_sha256": sha256_file(data_yaml_path),
        "class_count": len(class_names),
        "class_names": class_names,
        "splits": split_reports,
        "totals": {
            "images": sum(report["images"] for report in split_reports.values()),
            "labels": sum(report["labels"] for report in split_reports.values()),
            "instances": total_instances,
            "background_images": total_background_images,
            "missing_labels": total_missing_labels,
            "orphan_labels": total_orphan_labels,
            "invalid_rows": total_invalid_rows,
            "invalid_class_ids": total_invalid_class_ids,
            "invalid_coordinates": total_invalid_coordinates,
            "out_of_bounds_boxes": total_out_of_bounds_boxes,
            "zero_area_boxes": total_zero_area_boxes,
            "duplicate_boxes": total_duplicate_boxes,
            "corrupt_images": total_corrupt_images,
        },
        "classes": class_rows,
        "common_image_dimensions": [
            {
                "dimensions": dimensions,
                "count": count,
            }
            for dimensions, count in all_image_dimensions.most_common(20)
        ],
        "issue_count": len(all_issues),
        "issues": all_issues,
    }

    report_path = manifest_directory / "ppe_dataset_audit.json"

    class_csv_path = manifest_directory / "ppe_class_counts.csv"

    report_path.write_text(
        json.dumps(
            audit_report,
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    with class_csv_path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as file_handle:
        writer = csv.DictWriter(
            file_handle,
            fieldnames=[
                "class_id",
                "class_name",
                "instance_count",
                "image_count",
            ],
        )

        writer.writeheader()
        writer.writerows(class_rows)

    print()
    print("=" * 72)
    print("PPE DATASET AUDIT")
    print("=" * 72)
    print(f"Status: {audit_status}")
    print(f"Dataset: {dataset_root}")
    print(f"Classes: {len(class_names)}")

    for split_name, split_report in split_reports.items():
        print(
            f"{split_name}: "
            f"{split_report['images']} images / "
            f"{split_report['labels']} labels / "
            f"{split_report['instances']} instances / "
            f"{split_report['background_images']} backgrounds"
        )

    print(f"Total instances: {total_instances}")
    print(f"Missing labels: {total_missing_labels}")
    print(f"Orphan labels: {total_orphan_labels}")
    print(f"Invalid rows: {total_invalid_rows}")
    print(f"Invalid class IDs: {total_invalid_class_ids}")
    print(f"Invalid coordinates: {total_invalid_coordinates}")
    print(f"Out-of-bounds boxes: {total_out_of_bounds_boxes}")
    print(f"Zero-area boxes: {total_zero_area_boxes}")
    print(f"Duplicate boxes: {total_duplicate_boxes}")
    print(f"Corrupt images: {total_corrupt_images}")
    print(f"Report: {report_path}")
    print(f"Class counts: {class_csv_path}")
    print("=" * 72)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
