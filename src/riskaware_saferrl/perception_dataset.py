from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import yaml

DATASET_ENV_VAR = "RISK_AWARE_PPE_DATASET_ROOT"

DEFAULT_DATASET_DIRECTORY_NAME = "Personal Protective Equipment - Combined Model.v1i.yolov8"

EXPECTED_DATASET_CLASS_NAMES = (
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
)

CANONICAL_CLASS_METADATA: dict[str, dict[str, Any]] = {
    "Fall-Detected": {
        "canonical_name": "fall_detected",
        "display_name": "Fall Detected",
        "semantic_category": "safety_event",
        "entity_type": "event",
        "semantic_map_channel": "fall_events",
        "positive_or_negative_status": "alert",
        "safety_priority": "critical",
        "compatible_person_region": "full_body",
        "conflict_class": None,
    },
    "Gloves": {
        "canonical_name": "gloves",
        "display_name": "Gloves",
        "semantic_category": "ppe",
        "entity_type": "ppe_item",
        "semantic_map_channel": "ppe_compliance",
        "positive_or_negative_status": "positive",
        "safety_priority": "high",
        "compatible_person_region": "hands",
        "conflict_class": "no_gloves",
    },
    "Goggles": {
        "canonical_name": "goggles",
        "display_name": "Goggles",
        "semantic_category": "ppe",
        "entity_type": "ppe_item",
        "semantic_map_channel": "ppe_compliance",
        "positive_or_negative_status": "positive",
        "safety_priority": "high",
        "compatible_person_region": "face",
        "conflict_class": "no_goggles",
    },
    "Hardhat": {
        "canonical_name": "hardhat",
        "display_name": "Hardhat",
        "semantic_category": "ppe",
        "entity_type": "ppe_item",
        "semantic_map_channel": "ppe_compliance",
        "positive_or_negative_status": "positive",
        "safety_priority": "critical",
        "compatible_person_region": "head",
        "conflict_class": "no_hardhat",
    },
    "Ladder": {
        "canonical_name": "ladder",
        "display_name": "Ladder",
        "semantic_category": "equipment",
        "entity_type": "equipment",
        "semantic_map_channel": "hazards",
        "positive_or_negative_status": "neutral",
        "safety_priority": "medium",
        "compatible_person_region": "scene",
        "conflict_class": None,
    },
    "Mask": {
        "canonical_name": "mask",
        "display_name": "Mask",
        "semantic_category": "ppe",
        "entity_type": "ppe_item",
        "semantic_map_channel": "ppe_compliance",
        "positive_or_negative_status": "positive",
        "safety_priority": "medium",
        "compatible_person_region": "face",
        "conflict_class": "no_mask",
    },
    "NO-Gloves": {
        "canonical_name": "no_gloves",
        "display_name": "No Gloves",
        "semantic_category": "ppe_violation",
        "entity_type": "ppe_status",
        "semantic_map_channel": "ppe_violations",
        "positive_or_negative_status": "negative",
        "safety_priority": "high",
        "compatible_person_region": "hands",
        "conflict_class": "gloves",
    },
    "NO-Goggles": {
        "canonical_name": "no_goggles",
        "display_name": "No Goggles",
        "semantic_category": "ppe_violation",
        "entity_type": "ppe_status",
        "semantic_map_channel": "ppe_violations",
        "positive_or_negative_status": "negative",
        "safety_priority": "high",
        "compatible_person_region": "face",
        "conflict_class": "goggles",
    },
    "NO-Hardhat": {
        "canonical_name": "no_hardhat",
        "display_name": "No Hardhat",
        "semantic_category": "ppe_violation",
        "entity_type": "ppe_status",
        "semantic_map_channel": "ppe_violations",
        "positive_or_negative_status": "negative",
        "safety_priority": "critical",
        "compatible_person_region": "head",
        "conflict_class": "hardhat",
    },
    "NO-Mask": {
        "canonical_name": "no_mask",
        "display_name": "No Mask",
        "semantic_category": "ppe_violation",
        "entity_type": "ppe_status",
        "semantic_map_channel": "ppe_violations",
        "positive_or_negative_status": "negative",
        "safety_priority": "medium",
        "compatible_person_region": "face",
        "conflict_class": "mask",
    },
    "NO-Safety Vest": {
        "canonical_name": "no_safety_vest",
        "display_name": "No Safety Vest",
        "semantic_category": "ppe_violation",
        "entity_type": "ppe_status",
        "semantic_map_channel": "ppe_violations",
        "positive_or_negative_status": "negative",
        "safety_priority": "critical",
        "compatible_person_region": "torso",
        "conflict_class": "safety_vest",
    },
    "Person": {
        "canonical_name": "person",
        "display_name": "Person",
        "semantic_category": "worker",
        "entity_type": "worker",
        "semantic_map_channel": "workers",
        "positive_or_negative_status": "neutral",
        "safety_priority": "critical",
        "compatible_person_region": "full_body",
        "conflict_class": None,
    },
    "Safety Cone": {
        "canonical_name": "safety_cone",
        "display_name": "Safety Cone",
        "semantic_category": "safety_marker",
        "entity_type": "equipment",
        "semantic_map_channel": "hazards",
        "positive_or_negative_status": "neutral",
        "safety_priority": "medium",
        "compatible_person_region": "scene",
        "conflict_class": None,
    },
    "Safety Vest": {
        "canonical_name": "safety_vest",
        "display_name": "Safety Vest",
        "semantic_category": "ppe",
        "entity_type": "ppe_item",
        "semantic_map_channel": "ppe_compliance",
        "positive_or_negative_status": "positive",
        "safety_priority": "critical",
        "compatible_person_region": "torso",
        "conflict_class": "no_safety_vest",
    },
}


@dataclass(frozen=True)
class PpeDatasetLayout:
    root: Path
    data_yaml: Path
    train_images: Path
    train_labels: Path
    valid_images: Path
    valid_labels: Path
    test_images: Path
    test_labels: Path

    def as_serializable_dict(self) -> dict[str, str]:
        return {key: str(value) for key, value in asdict(self).items()}


def find_project_root(start: Path | None = None) -> Path:
    candidate = (start or Path(__file__)).resolve()

    if candidate.is_file():
        candidate = candidate.parent

    for directory in (candidate, *candidate.parents):
        source_package = directory / "src" / "riskaware_saferrl"

        if (directory / "pyproject.toml").is_file() and source_package.is_dir():
            return directory

    raise RuntimeError(f"Could not find the repository root from {candidate}.")


def resolve_ppe_dataset_root(
    *,
    project_root: Path | None = None,
    explicit_root: Path | None = None,
) -> Path:
    resolved_project_root = (
        project_root.resolve() if project_root is not None else find_project_root()
    )

    if explicit_root is not None:
        candidate = explicit_root
    else:
        environment_value = os.environ.get(DATASET_ENV_VAR)

        candidate = (
            Path(environment_value)
            if environment_value
            else (resolved_project_root / DEFAULT_DATASET_DIRECTORY_NAME)
        )

    candidate = candidate.expanduser().resolve()

    if not candidate.is_dir():
        raise FileNotFoundError(f"PPE dataset root was not found: {candidate}")

    return candidate


def normalize_class_names(
    raw_names: Any,
) -> list[str]:
    if isinstance(raw_names, list):
        return [str(value) for value in raw_names]

    if isinstance(raw_names, dict):
        indexed_names = sorted((int(key), str(value)) for key, value in raw_names.items())

        actual_ids = [class_id for class_id, _ in indexed_names]

        expected_ids = list(range(len(indexed_names)))

        if actual_ids != expected_ids:
            raise ValueError(f"Dataset class IDs must be contiguous from zero. Found: {actual_ids}")

        return [value for _, value in indexed_names]

    raise TypeError("data.yaml names must be a list or an integer-keyed mapping.")


def load_data_yaml(
    data_yaml: Path,
) -> dict[str, Any]:
    if not data_yaml.is_file():
        raise FileNotFoundError(f"data.yaml was not found: {data_yaml}")

    with data_yaml.open(
        "r",
        encoding="utf-8",
    ) as file_handle:
        loaded = yaml.safe_load(file_handle)

    if not isinstance(loaded, dict):
        raise TypeError("data.yaml must contain a mapping.")

    return loaded


def validate_class_order(
    class_names: list[str],
) -> None:
    actual = tuple(class_names)

    if actual != EXPECTED_DATASET_CLASS_NAMES:
        raise ValueError(
            "Unexpected PPE class order.\n"
            f"Expected: {EXPECTED_DATASET_CLASS_NAMES}\n"
            f"Actual:   {actual}"
        )


def validate_dataset_layout(
    dataset_root: Path,
) -> PpeDatasetLayout:
    root = dataset_root.resolve()

    layout = PpeDatasetLayout(
        root=root,
        data_yaml=root / "data.yaml",
        train_images=root / "train" / "images",
        train_labels=root / "train" / "labels",
        valid_images=root / "valid" / "images",
        valid_labels=root / "valid" / "labels",
        test_images=root / "test" / "images",
        test_labels=root / "test" / "labels",
    )

    required_paths = (
        layout.data_yaml,
        layout.train_images,
        layout.train_labels,
        layout.valid_images,
        layout.valid_labels,
        layout.test_images,
        layout.test_labels,
    )

    missing_paths = [path for path in required_paths if not path.exists()]

    if missing_paths:
        formatted_paths = "\n".join(f"- {path}" for path in missing_paths)

        raise FileNotFoundError(f"The PPE dataset layout is incomplete:\n{formatted_paths}")

    data_yaml = load_data_yaml(layout.data_yaml)

    class_names = normalize_class_names(data_yaml.get("names"))

    declared_class_count = data_yaml.get("nc")

    if declared_class_count is not None and int(declared_class_count) != len(class_names):
        raise ValueError(
            f"data.yaml class-count mismatch: nc={declared_class_count}, names={len(class_names)}"
        )

    validate_class_order(class_names)

    return layout


def build_class_schema(
    class_names: list[str],
) -> dict[str, Any]:
    validate_class_order(class_names)

    classes: list[dict[str, Any]] = []

    for class_id, dataset_name in enumerate(class_names):
        metadata = CANONICAL_CLASS_METADATA[dataset_name]

        classes.append(
            {
                "dataset_class_id": class_id,
                "dataset_class_name": dataset_name,
                **metadata,
            }
        )

    canonical_names = [item["canonical_name"] for item in classes]

    if len(canonical_names) != len(set(canonical_names)):
        raise ValueError("Canonical class names must be unique.")

    return {
        "schema_version": 1,
        "dataset_class_count": len(classes),
        "dataset_class_order_locked": True,
        "classes": classes,
    }


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as file_handle:
        for chunk in iter(
            lambda: file_handle.read(1024 * 1024),
            b"",
        ):
            digest.update(chunk)

    return digest.hexdigest()


def load_audit_report(
    path: Path,
) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Dataset audit report was not found: {path}")

    report = json.loads(path.read_text(encoding="utf-8"))

    if report.get("status") != "PASS":
        raise ValueError(
            "Dataset audit status must be PASS before training configuration is created."
        )

    report_class_names = report.get("class_names")

    if report_class_names is not None:
        validate_class_order([str(name) for name in report_class_names])

    return report
