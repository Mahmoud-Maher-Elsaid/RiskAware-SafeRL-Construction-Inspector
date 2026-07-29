from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"Required runtime log was not created: {path}")

    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"Invalid JSONL at {path}:{line_number}") from error
        if not isinstance(payload, dict):
            raise TypeError(f"Expected an object at {path}:{line_number}")
        records.append(payload)

    if not records:
        raise ValueError(f"Runtime log is empty: {path}")
    return records


def find_last(
    records: list[dict[str, Any]],
    event: str,
) -> dict[str, Any]:
    matches = [record for record in records if record.get("event") == event]
    if not matches:
        raise ValueError(f"Missing runtime event: {event}")
    return matches[-1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--project",
        type=Path,
        required=True,
    )
    args = parser.parse_args()

    project = args.project.resolve()
    manual_path = project / "webots" / "logs" / "manual_robot_test.jsonl"
    supervisor_path = project / "webots" / "logs" / "scenario_supervisor.jsonl"

    manual_records = read_jsonl(manual_path)
    supervisor_records = read_jsonl(supervisor_path)

    run_complete = find_last(
        manual_records,
        "run_complete",
    )
    reset_verification = find_last(
        supervisor_records,
        "reset_verification",
    )

    valid_sensor_samples = int(
        run_complete.get(
            "valid_sensor_samples",
            0,
        )
    )
    safe_stop_applied = bool(
        run_complete.get(
            "safe_stop_applied",
            False,
        )
    )
    reset_ok = bool(
        reset_verification.get(
            "reset_ok",
            False,
        )
    )

    if valid_sensor_samples < 1:
        raise RuntimeError("No valid robot sensor samples were recorded.")
    if not safe_stop_applied:
        raise RuntimeError("The robot safe-stop record is false.")
    if not reset_ok:
        raise RuntimeError("The Supervisor reset verification failed.")

    summary = {
        "manual_record_count": len(manual_records),
        "supervisor_record_count": len(supervisor_records),
        "robot_steps": int(run_complete.get("steps", 0)),
        "valid_sensor_samples": (valid_sensor_samples),
        "safe_stop_applied": safe_stop_applied,
        "reset_position_ok": bool(
            reset_verification.get(
                "position_ok",
                False,
            )
        ),
        "reset_rotation_ok": bool(
            reset_verification.get(
                "rotation_ok",
                False,
            )
        ),
        "reset_verified": reset_ok,
    }
    print(
        json.dumps(
            summary,
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
