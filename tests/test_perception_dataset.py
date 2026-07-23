from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from riskaware_saferrl.perception_dataset import (
    DATASET_ENV_VAR,
    EXPECTED_DATASET_CLASS_NAMES,
    build_class_schema,
    normalize_class_names,
    resolve_ppe_dataset_root,
    validate_class_order,
    validate_dataset_layout,
)


def create_dataset(root: Path) -> Path:
    for relative_path in (
        "train/images",
        "train/labels",
        "valid/images",
        "valid/labels",
        "test/images",
        "test/labels",
    ):
        (root / relative_path).mkdir(
            parents=True,
            exist_ok=True,
        )

    data_yaml = {
        "nc": len(EXPECTED_DATASET_CLASS_NAMES),
        "names": list(EXPECTED_DATASET_CLASS_NAMES),
        "train": "train/images",
        "val": "valid/images",
        "test": "test/images",
    }

    (root / "data.yaml").write_text(
        yaml.safe_dump(
            data_yaml,
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    return root


def test_normalize_class_names_accepts_mapping() -> None:
    raw_names = {index: name for index, name in enumerate(EXPECTED_DATASET_CLASS_NAMES)}

    assert normalize_class_names(raw_names) == list(EXPECTED_DATASET_CLASS_NAMES)


def test_validate_dataset_layout_accepts_structure(
    tmp_path: Path,
) -> None:
    dataset_root = create_dataset(tmp_path / "dataset")

    layout = validate_dataset_layout(dataset_root)

    assert layout.root == dataset_root.resolve()
    assert layout.data_yaml.is_file()


def test_validate_class_order_rejects_reordering() -> None:
    names = list(EXPECTED_DATASET_CLASS_NAMES)

    names[0], names[1] = names[1], names[0]

    with pytest.raises(
        ValueError,
        match="Unexpected PPE class order",
    ):
        validate_class_order(names)


def test_resolve_dataset_root_uses_environment(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project_root = tmp_path / "project"
    project_root.mkdir()

    dataset_root = create_dataset(tmp_path / "external_dataset")

    monkeypatch.setenv(
        DATASET_ENV_VAR,
        str(dataset_root),
    )

    resolved = resolve_ppe_dataset_root(project_root=project_root)

    assert resolved == dataset_root.resolve()


def test_class_schema_has_locked_unique_mapping() -> None:
    schema = build_class_schema(list(EXPECTED_DATASET_CLASS_NAMES))

    assert schema["dataset_class_count"] == 14
    assert schema["dataset_class_order_locked"] is True

    classes = schema["classes"]

    canonical_names = {item["canonical_name"] for item in classes}

    assert len(canonical_names) == 14
    assert classes[0]["dataset_class_name"] == ("Fall-Detected")
    assert classes[10]["canonical_name"] == ("no_safety_vest")
    assert classes[11]["entity_type"] == ("worker")

    json.dumps(schema)
