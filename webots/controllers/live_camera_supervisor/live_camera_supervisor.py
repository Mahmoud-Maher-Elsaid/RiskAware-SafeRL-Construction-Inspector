from __future__ import annotations

import os
from pathlib import Path

from controller import Supervisor

MAXIMUM_SIMULATION_TIME_SECONDS = 90.0


def project_root() -> Path:
    value = os.environ.get("RISK_AWARE_PROJECT_ROOT")

    if not value:
        raise RuntimeError("RISK_AWARE_PROJECT_ROOT is not set.")

    return Path(value).resolve()


def count_records(
    telemetry_path: Path,
) -> int:
    if not telemetry_path.is_file():
        return 0

    with telemetry_path.open(
        "r",
        encoding="utf-8",
    ) as telemetry_file:
        return sum(1 for line in telemetry_file if line.strip())


def main() -> None:
    supervisor = Supervisor()

    time_step = int(supervisor.getBasicTimeStep())

    output_directory = project_root() / "webots" / "logs" / "stage5a_live_camera"

    telemetry_path = output_directory / "stage5a_camera_frames.jsonl"

    completion_marker_path = output_directory / "stage5a_complete.marker"

    failure_marker_path = output_directory / "stage5a_failure.marker"

    timeout_marker_path = output_directory / "stage5a_timeout.marker"

    print(
        "STAGE5A_SUPERVISOR_READY",
        flush=True,
    )

    while supervisor.step(time_step) != -1:
        simulation_time = float(supervisor.getTime())

        captured = count_records(telemetry_path)

        supervisor.setLabel(
            0,
            "Stage 5A Live RGB Acquisition",
            0.018,
            0.025,
            0.038,
            0x071521,
            0.0,
            "Arial",
        )

        supervisor.setLabel(
            1,
            f"Captured frames: {captured}/120",
            0.018,
            0.07,
            0.025,
            0x006D7D,
            0.0,
            "Arial",
        )

        supervisor.setLabel(
            2,
            "CV inference: DISABLED",
            0.018,
            0.105,
            0.021,
            0xA84300,
            0.0,
            "Arial",
        )

        supervisor.setLabel(
            3,
            "Policy motor control: DISABLED",
            0.018,
            0.135,
            0.021,
            0x9A001C,
            0.0,
            "Arial",
        )

        if failure_marker_path.is_file():
            print(
                "STAGE5A_SUPERVISOR_FAILURE_MARKER",
                flush=True,
            )

            supervisor.simulationQuit(1)
            return

        if completion_marker_path.is_file():
            print(
                f"STAGE5A_SUPERVISOR_COMPLETE captured={captured}",
                flush=True,
            )

            supervisor.simulationQuit(0)
            return

        if simulation_time <= MAXIMUM_SIMULATION_TIME_SECONDS:
            continue

        timeout_marker_path.write_text(
            "STAGE5A_TIMEOUT\n",
            encoding="utf-8",
            newline="\n",
        )

        print(
            "STAGE5A_SUPERVISOR_TIMEOUT",
            flush=True,
        )

        supervisor.simulationQuit(2)
        return


if __name__ == "__main__":
    main()
