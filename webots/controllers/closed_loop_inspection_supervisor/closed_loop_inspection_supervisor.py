from __future__ import annotations

import json
import math
import os
from pathlib import Path

from controller import Supervisor

MAXIMUM_SIMULATION_TIME_SECONDS = 485.0
ROBOT_DEF = "SHOWCASE_ROBOT"
VIEWPOINT_DEF = "HUMAN_VIEWPOINT"
EYE_HEIGHT_METERS = 1.55
FORWARD_OFFSET_METERS = 0.35
YAW_SMOOTHING_ALPHA = 0.45
FINAL_VIEW_SECONDS = 3.0


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
        line
        for line in telemetry_path.read_text(
            encoding="utf-8",
            errors="replace",
        ).splitlines()
        if line.strip()
    ]

    if not lines:
        return None

    return json.loads(lines[-1])


def read_perception_status(
    root: Path,
) -> tuple[str, str]:
    summary_path = root / "perception_summary.json"
    ready_path = root / "perception_ready.json"

    if summary_path.is_file():
        payload = json.loads(
            summary_path.read_text(
                encoding="utf-8",
                errors="replace",
            )
        )

        detections = int(
            payload.get(
                "total_detections",
                0,
            )
        )

        return (
            "LIVE CUDA",
            f"Detections: {detections}",
        )

    if ready_path.is_file():
        payload = json.loads(
            ready_path.read_text(
                encoding="utf-8",
                errors="replace",
            )
        )

        device = str(
            payload.get(
                "device",
                "unknown",
            )
        ).upper()

        return (
            f"LIVE {device}",
            "Inference active",
        )

    return (
        "STARTING",
        "Waiting for perception",
    )


def normalize_angle(angle: float) -> float:
    return math.atan2(
        math.sin(angle),
        math.cos(angle),
    )


def smoothed_angle(
    current: float | None,
    target: float,
) -> float:
    if current is None:
        return target

    delta = normalize_angle(target - current)

    return normalize_angle(current + YAW_SMOOTHING_ALPHA * delta)


def horizontal_forward(
    orientation: list[float],
) -> tuple[float, float]:
    if len(orientation) != 9:
        raise RuntimeError("Robot orientation must contain nine values.")

    forward_x = float(orientation[0])
    forward_z = float(orientation[6])
    magnitude = math.hypot(
        forward_x,
        forward_z,
    )

    if not math.isfinite(magnitude) or magnitude < 1e-8:
        raise RuntimeError("Robot forward direction is invalid.")

    return (
        forward_x / magnitude,
        forward_z / magnitude,
    )


def viewpoint_yaw_from_forward(
    forward_x: float,
    forward_z: float,
) -> float:
    return math.atan2(
        -forward_z,
        forward_x,
    )


def update_human_view(
    robot_node: object,
    position_field: object,
    orientation_field: object,
    current_yaw: float | None,
) -> float:
    robot_position = robot_node.getPosition()
    robot_orientation = robot_node.getOrientation()

    forward_x, forward_z = horizontal_forward(robot_orientation)

    target_yaw = viewpoint_yaw_from_forward(
        forward_x,
        forward_z,
    )

    resolved_yaw = smoothed_angle(
        current_yaw,
        target_yaw,
    )

    position_field.setSFVec3f(
        [
            float(robot_position[0]) + FORWARD_OFFSET_METERS * forward_x,
            float(robot_position[1]) + EYE_HEIGHT_METERS,
            float(robot_position[2]) + FORWARD_OFFSET_METERS * forward_z,
        ]
    )

    orientation_field.setSFRotation(
        [
            0.0,
            1.0,
            0.0,
            resolved_yaw,
        ]
    )

    return resolved_yaw


def export_view(
    supervisor: Supervisor,
    output_directory: Path,
    filename: str,
) -> None:
    supervisor.exportImage(
        str(output_directory / filename),
        100,
    )


def main() -> None:
    supervisor = Supervisor()
    time_step = int(supervisor.getBasicTimeStep())

    root = project_root()
    mission_output = root / "webots" / "logs" / "stage5a3_closed_loop"

    perception_output = root / "webots" / "logs" / "stage5b3_live_perception"

    mission_output.mkdir(
        parents=True,
        exist_ok=True,
    )

    telemetry_path = mission_output / "stage5a3_mission_telemetry.jsonl"

    completion_marker = mission_output / "stage5a3_complete.marker"

    failure_marker = mission_output / "stage5a3_failure.marker"

    timeout_marker = mission_output / "stage5a3_timeout.marker"

    robot_node = supervisor.getFromDef(ROBOT_DEF)

    viewpoint_node = supervisor.getFromDef(VIEWPOINT_DEF)

    if robot_node is None:
        raise RuntimeError(f"Robot DEF was not found: {ROBOT_DEF}")

    if viewpoint_node is None:
        raise RuntimeError(f"Viewpoint DEF was not found: {VIEWPOINT_DEF}")

    position_field = viewpoint_node.getField("position")

    orientation_field = viewpoint_node.getField("orientation")

    if position_field is None or orientation_field is None:
        raise RuntimeError("Human Viewpoint fields are unavailable.")

    robot_node.enablePoseTracking(
        time_step,
        None,
    )

    print(
        "STAGE5A3_SUPERVISOR_READY",
        flush=True,
    )

    print(
        "HUMAN_FIRST_PERSON_VIEW_READY",
        flush=True,
    )

    current_yaw: float | None = None
    initial_exported = False
    middle_exported = False
    completion_seen_at: float | None = None

    while supervisor.step(time_step) != -1:
        simulation_time = float(supervisor.getTime())

        current_yaw = update_human_view(
            robot_node,
            position_field,
            orientation_field,
            current_yaw,
        )

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

        cv_status, cv_detail = read_perception_status(perception_output)

        supervisor.setLabel(
            0,
            "RiskAware SafeRL Human First-Person Inspection",
            0.018,
            0.022,
            0.034,
            0x071521,
            0.0,
            "Arial",
        )

        supervisor.setLabel(
            1,
            "View: LEVEL HUMAN EYE | Motion: AUTONOMOUS",
            0.018,
            0.064,
            0.021,
            0x126B37,
            0.0,
            "Arial",
        )

        supervisor.setLabel(
            2,
            f"State: {state} | Target: {target}",
            0.018,
            0.095,
            0.020,
            0x006D7D,
            0.0,
            "Arial",
        )

        supervisor.setLabel(
            3,
            (f"Frames: {frames} | Distance: {distance:.2f} m"),
            0.018,
            0.125,
            0.019,
            0x15202A,
            0.0,
            "Arial",
        )

        supervisor.setLabel(
            4,
            f"CV: {cv_status} | {cv_detail}",
            0.018,
            0.154,
            0.019,
            0xA84300,
            0.0,
            "Arial",
        )

        supervisor.setLabel(
            5,
            ("Motor source: CLOSED-LOOP WAYPOINT | RL motor gate: DISABLED"),
            0.018,
            0.183,
            0.018,
            0x9A001C,
            0.0,
            "Arial",
        )

        if not initial_exported and simulation_time >= 4.0:
            export_view(
                supervisor,
                mission_output,
                "first_person_initial.jpg",
            )

            initial_exported = True

        if not middle_exported and frames >= 700:
            export_view(
                supervisor,
                mission_output,
                "first_person_middle.jpg",
            )

            middle_exported = True

        if failure_marker.is_file():
            export_view(
                supervisor,
                mission_output,
                "first_person_failure.jpg",
            )

            print(
                "STAGE5A3_SUPERVISOR_FAILURE_MARKER",
                flush=True,
            )

            supervisor.simulationQuit(1)
            return

        if completion_marker.is_file():
            if completion_seen_at is None:
                completion_seen_at = simulation_time

                export_view(
                    supervisor,
                    mission_output,
                    "first_person_final.jpg",
                )

                print(
                    "HUMAN_FIRST_PERSON_MISSION_COMPLETE",
                    flush=True,
                )

            if simulation_time - completion_seen_at >= FINAL_VIEW_SECONDS:
                print(
                    (f"STAGE5A3_SUPERVISOR_COMPLETE frames={frames}"),
                    flush=True,
                )

                supervisor.simulationQuit(0)
                return

            continue

        if simulation_time <= MAXIMUM_SIMULATION_TIME_SECONDS:
            continue

        timeout_marker.write_text(
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
