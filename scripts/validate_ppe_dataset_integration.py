from __future__ import annotations

import json
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"

if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from riskaware_saferrl.perception_dataset import (  # noqa: E402
    EXPECTED_DATASET_CLASS_NAMES,
    build_class_schema,
    load_audit_report,
    load_data_yaml,
    normalize_class_names,
    resolve_ppe_dataset_root,
    sha256_file,
    validate_dataset_layout,
)


def load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Required JSON file was not found: {path}")

    payload = json.loads(path.read_text(encoding="utf-8"))

    if not isinstance(payload, dict):
        raise TypeError(f"Expected a JSON object in: {path}")

    return payload


def load_yaml(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Required YAML file was not found: {path}")

    payload = yaml.safe_load(path.read_text(encoding="utf-8"))

    if not isinstance(payload, dict):
        raise TypeError(f"Expected a YAML mapping in: {path}")

    return payload


def verify_git_ignore(project_root: Path, dataset_root: Path) -> bool:
    relative_path = dataset_root.relative_to(project_root)

    completed = subprocess.run(
        [
            "git",
            "check-ignore",
            "--quiet",
            "--",
            str(relative_path),
        ],
        cwd=project_root,
        check=False,
    )

    return completed.returncode == 0


def write_report(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def main() -> int:
    dataset_root = resolve_ppe_dataset_root(project_root=PROJECT_ROOT)
    layout = validate_dataset_layout(dataset_root)

    audit_path = PROJECT_ROOT / "data" / "manifests" / "ppe_dataset_audit.json"

    manifest_path = PROJECT_ROOT / "data" / "manifests" / "ppe_dataset_manifest.json"

    schema_path = PROJECT_ROOT / "configs" / "perception" / "class_schema.json"

    config_path = PROJECT_ROOT / "configs" / "perception" / "ppe_dataset.yaml"

    audit = load_audit_report(audit_path)
    manifest = load_json(manifest_path)
    schema = load_json(schema_path)
    config = load_yaml(config_path)
    data_yaml = load_data_yaml(layout.data_yaml)

    class_names = normalize_class_names(data_yaml.get("names"))
    expected_names = list(EXPECTED_DATASET_CLASS_NAMES)

    failures: list[str] = []

    if class_names != expected_names:
        failures.append("data.yaml class order does not match the locked schema.")

    if manifest.get("class_names") != expected_names:
        failures.append("Dataset manifest class order does not match data.yaml.")

    if manifest.get("class_count") != 14:
        failures.append("Dataset manifest class count is not 14.")

    if manifest.get("audit_status") != "PASS":
        failures.append("Dataset manifest audit status is not PASS.")

    if audit.get("status") != "PASS":
        failures.append("Dataset audit status is not PASS.")

    if schema.get("dataset_class_count") != 14:
        failures.append("Class schema count is not 14.")

    if schema.get("dataset_class_order_locked") is not True:
        failures.append("Class schema order is not locked.")

    classes = schema.get("classes")

    if not isinstance(classes, list) or len(classes) != 14:
        failures.append("Class schema does not contain exactly 14 classes.")
    else:
        schema_dataset_names = [str(item.get("dataset_class_name")) for item in classes]

        schema_class_ids = [item.get("dataset_class_id") for item in classes]

        canonical_names = [str(item.get("canonical_name")) for item in classes]

        if schema_dataset_names != expected_names:
            failures.append("Class schema dataset names are out of order.")

        if schema_class_ids != list(range(14)):
            failures.append("Class schema IDs are not contiguous from zero.")

        if len(canonical_names) != len(set(canonical_names)):
            failures.append("Canonical class names are not unique.")

    validation_config = config.get("validation", {})

    if validation_config.get("expected_class_count") != 14:
        failures.append("Dataset configuration expected class count is not 14.")

    if validation_config.get("lock_dataset_class_order") is not True:
        failures.append("Dataset configuration does not lock class order.")

    if manifest.get("data_yaml_sha256") != sha256_file(layout.data_yaml):
        failures.append("data.yaml SHA256 does not match the dataset manifest.")

    if manifest.get("audit_manifest_sha256") != sha256_file(audit_path):
        failures.append("Audit SHA256 does not match the dataset manifest.")

    if not verify_git_ignore(PROJECT_ROOT, dataset_root):
        failures.append("The local PPE dataset is not ignored by Git.")

    rebuilt_schema = build_class_schema(class_names)

    if rebuilt_schema != schema:
        failures.append("Stored class schema differs from the schema rebuilt from data.yaml.")

    split_names = ("train", "valid", "test")

    for split_name in split_names:
        split = audit.get("splits", {}).get(split_name, {})

        if split.get("images") != split.get("labels"):
            failures.append(f"{split_name} image and label counts do not match.")

        for issue_name in (
            "missing_labels",
            "orphan_labels",
            "invalid_rows",
            "invalid_class_ids",
            "invalid_coordinates",
            "out_of_bounds_boxes",
            "zero_area_boxes",
            "duplicate_boxes",
            "corrupt_images",
        ):
            if split.get(issue_name) != 0:
                failures.append(f"{split_name} has nonzero {issue_name}: {split.get(issue_name)}")

    report = {
        "schema_version": 1,
        "created_at": datetime.now(UTC).isoformat(),
        "status": "FAIL" if failures else "PASS",
        "project_root": str(PROJECT_ROOT),
        "dataset_root": str(dataset_root),
        "class_count": len(class_names),
        "class_names": class_names,
        "data_yaml_sha256": sha256_file(layout.data_yaml),
        "audit_manifest_sha256": sha256_file(audit_path),
        "dataset_git_ignored": verify_git_ignore(
            PROJECT_ROOT,
            dataset_root,
        ),
        "split_summary": audit.get("splits"),
        "totals": audit.get("totals"),
        "failures": failures,
    }

    report_path = PROJECT_ROOT / "data" / "manifests" / "ppe_dataset_integration_validation.json"

    write_report(report_path, report)

    if failures:
        print()
        print("=" * 72)
        print("PPE DATASET INTEGRATION VALIDATION FAILED")
        print("=" * 72)

        for failure in failures:
            print(f"- {failure}")

        print(f"Report: {report_path}")
        print("=" * 72)

        return 1

    print()
    print("=" * 72)
    print("PPE DATASET INTEGRATION VALIDATION PASSED")
    print("=" * 72)
    print(f"Dataset root: {dataset_root}")
    print(f"Class count: {len(class_names)}")
    print(f"Audit status: {audit['status']}")
    print(f"Total images: {audit['totals']['images']}")
    print(f"Total labels: {audit['totals']['labels']}")
    print(f"Total instances: {audit['totals']['instances']}")
    print(f"Report: {report_path}")
    print("=" * 72)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
