from __future__ import annotations

import json
import os
from pathlib import Path

from controller import Supervisor

MAXIMUM_SIMULATION_TIME_SECONDS = 485.0


def project_root() -> Path:
    value = os.environ.get("RISK_AWARE_PROJECT_ROOT")

    if not value:
        raise RuntimeError("RISK_AWARE_PROJECT_ROOT is not set.")

    return Path(value).resolve()


def read_last_record(
    telemetry_path: Path,
) -> dict[str, object] | None:
    if not telemetry_path.is_file():
        return None

    lines = [
        line for line in telemetry_path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]

    if not lines:
        return None

    return json.loads(lines[-1])


def main() -> None:
    supervisor = Supervisor()
    time_step = int(supervisor.getBasicTimeStep())

    output_directory = project_root() / "webots" / "logs" / "stage5a3_closed_loop"

    telemetry_path = output_directory / "stage5a3_mission_telemetry.jsonl"

    completion_marker_path = output_directory / "stage5a3_complete.marker"

    failure_marker_path = output_directory / "stage5a3_failure.marker"

    timeout_marker_path = output_directory / "stage5a3_timeout.marker"

    print(
        "STAGE5A3_SUPERVISOR_READY",
        flush=True,
    )

    while supervisor.step(time_step) != -1:
        simulation_time = float(supervisor.getTime())

        record = read_last_record(telemetry_path)

        if record is None:
            frames = 0
            state = "STARTING"
            target = "NONE"
            distance = float("nan")
        else:
            frames = int(record["capture_index"]) + 1
            state = str(record["mission_state"])
            target = str(record["target_waypoint_name"])
            distance = float(record["distance_to_target_meters"])

        supervisor.setLabel(
            0,
            "Stage 5A3 Closed-Loop Inspection Mission",
            0.018,
            0.022,
            0.036,
            0x071521,
            0.0,
            "Arial",
        )

        supervisor.setLabel(
            1,
            f"State: {state}",
            0.018,
            0.066,
            0.024,
            0x006D7D,
            0.0,
            "Arial",
        )

        supervisor.setLabel(
            2,
            f"Target: {target}",
            0.018,
            0.098,
            0.022,
            0xA84300,
            0.0,
            "Arial",
        )

        supervisor.setLabel(
            3,
            f"Frames: {frames} | Distance: {distance:.2f} m",
            0.018,
            0.128,
            0.021,
            0x15202A,
            0.0,
            "Arial",
        )

        supervisor.setLabel(
            4,
            "Motion: CLOSED-LOOP WAYPOINT CONTROLLER",
            0.018,
            0.158,
            0.020,
            0x126B37,
            0.0,
            "Arial",
        )

        supervisor.setLabel(
            5,
            "CV inference: DISABLED | Policy control: DISABLED",
            0.018,
            0.187,
            0.019,
            0x9A001C,
            0.0,
            "Arial",
        )

        if failure_marker_path.is_file():
            print(
                "STAGE5A3_SUPERVISOR_FAILURE_MARKER",
                flush=True,
            )

            supervisor.simulationQuit(1)
            return

        if completion_marker_path.is_file():
            print(
                f"STAGE5A3_SUPERVISOR_COMPLETE frames={frames}",
                flush=True,
            )

            supervisor.simulationQuit(0)
            return

        if simulation_time <= MAXIMUM_SIMULATION_TIME_SECONDS:
            continue

        timeout_marker_path.write_text(
            "STAGE5A3_TIMEOUT\n",
            encoding="utf-8",
            newline="\n",
        )

        print(
            "STAGE5A3_SUPERVISOR_TIMEOUT",
            flush=True,
        )

        supervisor.simulationQuit(2)
        return


if __name__ == "__main__":
    main()
