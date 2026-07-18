from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from riskaware_saferrl.webots import (
    GridAction,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]

ROBOT_LOG_PATH = PROJECT_ROOT / "webots" / "logs" / "stage4b2_live_robot.jsonl"

POLICY_LOG_PATH = PROJECT_ROOT / "webots" / "logs" / "stage4c_live_policy_proposals.jsonl"

SUMMARY_PATH = PROJECT_ROOT / "webots" / "logs" / "stage4c_live_policy_runtime_summary.json"

EXPECTED_SHA256 = "172437CAE45B69031F443C0707FB0795D2F1860D3B95594BE281645D8A173FE7"

MINIMUM_SAMPLE_COUNT = 30


def require(
    condition: bool,
    message: str,
) -> None:
    if not condition:
        raise RuntimeError(message)


def load_jsonl(
    path: Path,
) -> list[dict[str, Any]]:
    if not path.is_file():
        raise RuntimeError(f"Runtime log was not found: {path}")

    records: list[dict[str, Any]] = []

    for line_number, raw_line in enumerate(
        path.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        line = raw_line.strip()

        if not line:
            continue

        try:
            payload = json.loads(line)
        except json.JSONDecodeError as error:
            raise RuntimeError(f"Invalid JSON on line {line_number} of {path.name}.") from error

        if not isinstance(
            payload,
            dict,
        ):
            raise RuntimeError(f"Expected an object on line {line_number} of {path.name}.")

        records.append(payload)

    return records


def validate_policy_record(
    record: dict[str, Any],
) -> None:
    require(
        int(record["schema_version"]) == 1,
        "Unexpected policy proposal schema version.",
    )

    sample_index = int(record["sample_index"])

    require(
        sample_index >= 0,
        "Policy sample index must be non-negative.",
    )

    action = int(record["action"])

    require(
        0 <= action < 5,
        "Policy action index is invalid.",
    )

    require(
        str(record["action_name"]) == GridAction(action).name,
        "Policy action name does not match its index.",
    )

    action_mask = [bool(value) for value in record["action_mask"]]

    require(
        len(action_mask) == 5,
        "Policy action mask must contain five values.",
    )

    require(
        any(action_mask),
        "Policy action mask has no valid action.",
    )

    require(
        action_mask[action],
        "Policy selected a masked-out action.",
    )

    require(
        bool(record["mask_respected"]),
        "mask_respected is not true.",
    )

    require(
        not bool(record["motors_connected"]),
        "Policy proposal reports a motor connection.",
    )

    require(
        not bool(record["policy_controls_motors"]),
        "Policy controls the motors.",
    )

    require(
        not bool(record["policy_action_applied"]),
        "Policy action was applied.",
    )

    require(
        bool(record["proposal_only"]),
        "Policy record is not proposal-only.",
    )

    require(
        str(record["motor_command_channel"]) == "none",
        "Policy record exposes a motor command channel.",
    )

    require(
        not bool(record["robot_controller_modified"]),
        "Policy sidecar reports a modified robot controller.",
    )

    require(
        str(record["observation_source"]) == ("live_webots_telemetry_reconstruction"),
        "Unexpected observation source.",
    )

    require(
        bool(record["live_map_verified"]),
        "Live semantic map was not verified.",
    )

    require(
        bool(record["live_state_verified"]),
        "Live state was not verified.",
    )

    require(
        bool(record["live_action_mask_verified"]),
        "Live action mask was not verified.",
    )

    require(
        str(record["checkpoint_sha256"]) == EXPECTED_SHA256,
        "Policy checkpoint SHA256 is invalid.",
    )

    finite_values = (
        record["simulation_time"],
        record["world_x"],
        record["world_z"],
    )

    require(
        all(math.isfinite(float(value)) for value in finite_values),
        "Policy runtime record contains a non-finite value.",
    )


def strictly_increasing(
    values: list[int],
) -> bool:
    return all(
        current > previous
        for previous, current in zip(
            values,
            values[1:],
            strict=False,
        )
    )


def main() -> None:
    robot_records = load_jsonl(ROBOT_LOG_PATH)

    policy_records = load_jsonl(POLICY_LOG_PATH)

    require(
        len(robot_records) >= MINIMUM_SAMPLE_COUNT,
        "Robot log contains fewer than 30 samples.",
    )

    require(
        len(policy_records) >= MINIMUM_SAMPLE_COUNT,
        "Policy log contains fewer than 30 proposals.",
    )

    for record in policy_records:
        validate_policy_record(record)

    robot_by_index = {int(record["sample_index"]): record for record in robot_records}

    policy_by_index = {int(record["sample_index"]): record for record in policy_records}

    policy_indices = list(policy_by_index)

    require(
        strictly_increasing([int(record["sample_index"]) for record in policy_records]),
        "Policy sample indices are not strictly increasing.",
    )

    common_indices = set(robot_by_index) & set(policy_by_index)

    require(
        len(common_indices) >= MINIMUM_SAMPLE_COUNT,
        "Fewer than 30 live samples received policy proposals.",
    )

    for sample_index in sorted(common_indices):
        robot_record = robot_by_index[sample_index]

        policy_record = policy_by_index[sample_index]

        robot_mask = [bool(value) for value in robot_record["action_mask"]]

        policy_mask = [bool(value) for value in policy_record["action_mask"]]

        require(
            robot_mask == policy_mask,
            (f"Policy mask differs from the live robot mask at sample {sample_index}."),
        )

        require(
            int(robot_record["grid_row"]) == int(policy_record["grid_row"]),
            "Policy grid row differs from live telemetry.",
        )

        require(
            int(robot_record["grid_column"]) == int(policy_record["grid_column"]),
            "Policy grid column differs from live telemetry.",
        )

        require(
            math.isclose(
                float(robot_record["simulation_time"]),
                float(policy_record["simulation_time"]),
                rel_tol=0.0,
                abs_tol=1e-9,
            ),
            "Policy simulation time differs from live telemetry.",
        )

    unique_actions = sorted({int(record["action"]) for record in policy_records})

    summary = {
        "schema_version": 1,
        "robot_sample_count": len(robot_records),
        "policy_proposal_count": len(policy_records),
        "matched_live_sample_count": len(common_indices),
        "first_policy_sample_index": min(policy_indices),
        "last_policy_sample_index": max(policy_indices),
        "unique_proposed_actions": (unique_actions),
        "invalid_proposal_count": 0,
        "all_masks_respected": True,
        "live_map_verified": True,
        "live_state_verified": True,
        "live_action_mask_verified": True,
        "controller_unchanged": True,
        "motors_connected": False,
        "policy_actions_applied": False,
        "motor_isolation_verified": True,
        "live_policy_runtime_verified": True,
    }

    SUMMARY_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    SUMMARY_PATH.write_text(
        json.dumps(
            summary,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    print(
        json.dumps(
            summary,
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
