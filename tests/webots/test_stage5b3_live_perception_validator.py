from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
from types import ModuleType


def load_validator_module() -> ModuleType:
    project_root = Path(__file__).resolve().parents[2]
    script_path = project_root / "scripts" / "validate_stage5b3_live_perception_mission.py"
    specification = importlib.util.spec_from_file_location(
        "stage5b3_validator",
        script_path,
    )
    if specification is None or specification.loader is None:
        raise RuntimeError("Could not load the Stage 5B3 validator.")

    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def test_load_json_reads_object(tmp_path: Path) -> None:
    module = load_validator_module()
    path = tmp_path / "payload.json"
    path.write_text(
        json.dumps({"stage": "5B3", "runtime_verified": True}),
        encoding="utf-8",
    )
    payload = module.load_json(path)
    assert payload["stage"] == "5B3"
    assert payload["runtime_verified"] is True


def test_find_stage5a3_report_uses_latest(tmp_path: Path) -> None:
    module = load_validator_module()
    older = tmp_path / "older.json"
    newer = tmp_path / "newer.json"

    older.write_text(
        json.dumps(
            {
                "stage": "5A3",
                "runtime_verified": True,
                "route_completed": False,
            }
        ),
        encoding="utf-8",
    )
    newer.write_text(
        json.dumps(
            {
                "stage": "5A3",
                "runtime_verified": True,
                "route_completed": True,
            }
        ),
        encoding="utf-8",
    )

    os.utime(older, ns=(1_000_000_000, 1_000_000_000))
    os.utime(newer, ns=(2_000_000_000, 2_000_000_000))

    selected_path, payload = module.find_stage5a3_report(tmp_path)
    assert selected_path == newer
    assert payload["route_completed"] is True
