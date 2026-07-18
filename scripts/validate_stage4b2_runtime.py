from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]

ROBOT_LOG_PATH = PROJECT_ROOT / "webots" / "logs" / "stage4b2_live_robot.jsonl"

SUPERVISOR_LOG_PATH = PROJECT_ROOT / "webots" / "logs" / "stage4b2_live_supervisor.jsonl"

SUMMARY_PATH = PROJECT_ROOT / "webots" / "logs" / "stage4b2_runtime_summary.json"

MINIMUM_SAMPLE_COUNT = 30
MINIMUM_POSITION_SPAN = 0.10


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

        if not isinstance(payload, dict):
            raise RuntimeError(f"Expected an object on line {line_number} of {path.name}.")

        records.append(payload)

    return records


def require(
    condition: bool,
    message: str,
) -> None:
    if not condition:
        raise RuntimeError(message)


def require_finite_sequence(
    values: list[object],
    *,
    name: str,
) -> None:
    require(
        all(math.isfinite(float(value)) for value in values),
        f"{name} contains a non-finite value.",
    )


def validate_sample(
    record: dict[str, Any],
) -> None:
    require(
        int(record["schema_version"]) == 1,
        "Unexpected telemetry schema version.",
    )

    require(
        int(record["map_length"]) == 1008,
        "Telemetry map length must equal 1008.",
    )

    require(
        0 <= int(record["map_nonzero"]) <= 1008,
        "Telemetry map_nonzero is invalid.",
    )

    proximity = list(record["proximity"])

    compass = list(record["compass"])

    state = list(record["state"])

    action_mask = list(record["action_mask"])

    require(
        len(proximity) == 8,
        "Telemetry must contain eight proximity readings.",
    )

    require(
        len(compass) == 3,
        "Telemetry must contain three compass values.",
    )

    require(
        len(state) == 4,
        "Telemetry must contain a four-value state vector.",
    )

    require(
        len(action_mask) == 5,
        "Telemetry must contain a five-value action mask.",
    )

    require(
        any(bool(value) for value in action_mask),
        "Telemetry action mask has no valid action.",
    )

    require_finite_sequence(
        proximity,
        name="proximity",
    )

    require_finite_sequence(
        compass,
        name="compass",
    )

    require_finite_sequence(
        state,
        name="state",
    )

    require_finite_sequence(
        [
            record["simulation_time"],
            record["world_x"],
            record["world_z"],
            record["yaw_radians"],
            record["map_sum"],
        ],
        name="telemetry scalars",
    )

    require(
        all(0.0 <= float(value) <= 1.0 for value in state),
        "Telemetry state vector is not normalized.",
    )

    require(
        0 <= int(record["grid_row"]) < 12,
        "Telemetry grid row is invalid.",
    )

    require(
        0 <= int(record["grid_column"]) < 12,
        "Telemetry grid column is invalid.",
    )

    require(
        str(record["heading"])
        in {
            "EAST",
            "NORTH",
            "WEST",
            "SOUTH",
        },
        "Telemetry heading is invalid.",
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

    supervisor_records = load_jsonl(SUPERVISOR_LOG_PATH)

    require(
        len(robot_records) >= MINIMUM_SAMPLE_COUNT,
        "The robot log contains fewer than 30 telemetry samples.",
    )

    require(
        len(supervisor_records) >= MINIMUM_SAMPLE_COUNT,
        "The Supervisor log contains fewer than 30 telemetry samples.",
    )

    for record in robot_records:
        validate_sample(record)

    for record in supervisor_records:
        validate_sample(record)

        require(
            "supervisor_receive_time" in record,
            "Supervisor receive time is missing.",
        )

        require_finite_sequence(
            [record["supervisor_receive_time"]],
            name="Supervisor receive time",
        )

    robot_indices = [int(record["sample_index"]) for record in robot_records]

    supervisor_indices = [int(record["sample_index"]) for record in supervisor_records]

    require(
        strictly_increasing(robot_indices),
        "Robot sample indices are not strictly increasing.",
    )

    require(
        strictly_increasing(supervisor_indices),
        "Supervisor sample indices are not strictly increasing.",
    )

    common_indices = set(robot_indices) & set(supervisor_indices)

    require(
        len(common_indices) >= MINIMUM_SAMPLE_COUNT,
        "Fewer than 30 telemetry samples were transferred successfully.",
    )

    world_x_values = [float(record["world_x"]) for record in robot_records]

    world_z_values = [float(record["world_z"]) for record in robot_records]

    position_span = math.hypot(
        max(world_x_values) - min(world_x_values),
        max(world_z_values) - min(world_z_values),
    )

    require(
        position_span >= MINIMUM_POSITION_SPAN,
        "The live GPS position did not change enough during runtime.",
    )

    headings = {str(record["heading"]) for record in robot_records}

    require(
        len(headings) >= 2,
        "The live heading did not change during runtime.",
    )

    grid_positions = {
        (
            int(record["grid_row"]),
            int(record["grid_column"]),
        )
        for record in robot_records
    }

    maximum_ack_count = max(
        int(
            record.get(
                "ack_count",
                0,
            )
        )
        for record in robot_records
    )

    require(
        maximum_ack_count > 0,
        "The robot did not receive any Supervisor acknowledgement.",
    )

    receive_times = [float(record["supervisor_receive_time"]) for record in supervisor_records]

    require(
        all(
            current >= previous
            for previous, current in zip(
                receive_times,
                receive_times[1:],
                strict=False,
            )
        ),
        "Supervisor receive times are not ordered.",
    )

    compass_norms = [
        math.sqrt(sum(float(value) ** 2 for value in record["compass"])) for record in robot_records
    ]

    require(
        all(value > 0.01 for value in compass_norms),
        "The Compass returned an invalid vector.",
    )

    summary = {
        "schema_version": 1,
        "robot_sample_count": len(robot_records),
        "supervisor_sample_count": len(supervisor_records),
        "transferred_sample_count": len(common_indices),
        "first_sample_index": min(robot_indices),
        "last_sample_index": max(robot_indices),
        "world_position_span": position_span,
        "unique_heading_count": len(headings),
        "headings": sorted(headings),
        "unique_grid_position_count": len(grid_positions),
        "grid_positions": [list(position) for position in sorted(grid_positions)],
        "maximum_ack_count": (maximum_ack_count),
        "map_length_verified": True,
        "state_length_verified": True,
        "action_mask_verified": True,
        "proximity_sensor_count": 8,
        "compass_verified": True,
        "runtime_verified": True,
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
