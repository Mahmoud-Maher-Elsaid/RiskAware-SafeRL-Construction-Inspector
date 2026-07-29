from __future__ import annotations

import argparse
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
    DATASET_ENV_VAR,
    DEFAULT_DATASET_DIRECTORY_NAME,
    build_class_schema,
    find_project_root,
    load_audit_report,
    load_data_yaml,
    normalize_class_names,
    resolve_ppe_dataset_root,
    sha256_file,
    validate_dataset_layout,
)


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


def verify_git_ignore(
    project_root: Path,
    dataset_root: Path,
) -> bool:
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


def main() -> int:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--project-root",
        type=Path,
        default=None,
    )

    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=None,
    )

    arguments = parser.parse_args()

    project_root = (
        arguments.project_root.resolve()
        if arguments.project_root is not None
        else find_project_root(Path(__file__))
    )

    dataset_root = resolve_ppe_dataset_root(
        project_root=project_root,
        explicit_root=arguments.dataset_root,
    )

    layout = validate_dataset_layout(dataset_root)

    data_yaml = load_data_yaml(layout.data_yaml)

    class_names = normalize_class_names(data_yaml.get("names"))

    audit_path = project_root / "data" / "manifests" / "ppe_dataset_audit.json"

    audit_report = load_audit_report(audit_path)

    if not verify_git_ignore(
        project_root,
        dataset_root,
    ):
        raise RuntimeError("The local PPE dataset is not ignored by Git.")

    class_schema = build_class_schema(class_names)

    config_payload = {
        "schema_version": 1,
        "dataset": {
            "environment_variable": (DATASET_ENV_VAR),
            "default_relative_root": (DEFAULT_DATASET_DIRECTORY_NAME),
            "data_yaml": "data.yaml",
            "local_only": True,
            "git_tracked": False,
            "splits": {
                "train": {
                    "images": "train/images",
                    "labels": "train/labels",
                },
                "valid": {
                    "images": "valid/images",
                    "labels": "valid/labels",
                },
                "test": {
                    "images": "test/images",
                    "labels": "test/labels",
                },
            },
        },
        "validation": {
            "required_audit_status": "PASS",
            "audit_manifest": ("data/manifests/ppe_dataset_audit.json"),
            "class_schema": ("configs/perception/class_schema.json"),
            "expected_class_count": (len(class_names)),
            "lock_dataset_class_order": True,
        },
    }

    config_path = project_root / "configs" / "perception" / "ppe_dataset.yaml"

    config_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    config_path.write_text(
        yaml.safe_dump(
            config_payload,
            sort_keys=False,
            allow_unicode=True,
        ),
        encoding="utf-8",
    )

    class_schema_path = project_root / "configs" / "perception" / "class_schema.json"

    write_json(
        class_schema_path,
        class_schema,
    )

    manifest_payload = {
        "schema_version": 1,
        "created_at": datetime.now(UTC).isoformat(),
        "dataset_root_mode": "project_relative",
        "dataset_root_relative": str(dataset_root.relative_to(project_root)).replace("\\", "/"),
        "data_yaml_relative": str(layout.data_yaml.relative_to(project_root)).replace("\\", "/"),
        "data_yaml_sha256": sha256_file(layout.data_yaml),
        "audit_manifest_relative": str(audit_path.relative_to(project_root)).replace("\\", "/"),
        "audit_manifest_sha256": sha256_file(audit_path),
        "audit_status": audit_report["status"],
        "class_count": len(class_names),
        "class_names": class_names,
        "splits": audit_report["splits"],
        "totals": audit_report["totals"],
        "local_only": True,
        "git_ignored": True,
    }

    manifest_path = project_root / "data" / "manifests" / "ppe_dataset_manifest.json"

    write_json(
        manifest_path,
        manifest_payload,
    )

    print()
    print("=" * 72)
    print("PPE DATASET INTEGRATION COMPLETED")
    print("=" * 72)
    print(f"Project root: {project_root}")
    print(f"Dataset root: {dataset_root}")
    print(f"Class count: {len(class_names)}")
    print(f"Audit status: {audit_report['status']}")
    print(f"Dataset config: {config_path}")
    print(f"Class schema: {class_schema_path}")
    print(f"Dataset manifest: {manifest_path}")
    print("=" * 72)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
