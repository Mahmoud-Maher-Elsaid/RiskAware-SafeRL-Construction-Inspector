from __future__ import annotations

import math
import os
from pathlib import Path

from controller import Supervisor

EYE_HEIGHT_METERS = 1.55
FORWARD_OFFSET_METERS = 0.35
LEVELING_RADIANS = -math.pi / 2.0


def normalize_angle(angle: float) -> float:
    return math.atan2(math.sin(angle), math.cos(angle))


def viewpoint_orientation(yaw: float) -> list[float]:
    half_yaw = yaw / 2.0
    half_level = LEVELING_RADIANS / 2.0
    quaternion_w = math.cos(half_yaw) * math.cos(half_level)
    quaternion_x = math.cos(half_yaw) * math.sin(half_level)
    quaternion_y = math.sin(half_yaw) * math.cos(half_level)
    quaternion_z = -math.sin(half_yaw) * math.sin(half_level)
    vector_norm = math.sqrt(quaternion_x**2 + quaternion_y**2 + quaternion_z**2)
    angle = 2.0 * math.atan2(vector_norm, quaternion_w)
    return [
        quaternion_x / vector_norm,
        quaternion_y / vector_norm,
        quaternion_z / vector_norm,
        normalize_angle(angle),
    ]


def main() -> None:
    supervisor = Supervisor()
    time_step = int(supervisor.getBasicTimeStep())
    root = Path(os.environ["RISK_AWARE_PROJECT_ROOT"]).resolve()
    output = root / "webots" / "logs" / "stage5c_rl_motor_runtime"
    complete = output / "stage5c_complete.marker"
    failure = output / "stage5c_failure.json"
    robot = supervisor.getFromDef("SHOWCASE_ROBOT")
    viewpoint = supervisor.getFromDef("HUMAN_VIEWPOINT")
    if robot is None or viewpoint is None:
        raise RuntimeError("Stage 5C robot or human Viewpoint DEF is missing.")
    position_field = viewpoint.getField("position")
    orientation_field = viewpoint.getField("orientation")
    initial_exported = False
    completion_time: float | None = None

    while supervisor.step(time_step) != -1:
        position = robot.getPosition()
        orientation = robot.getOrientation()
        forward_x = float(orientation[0])
        forward_z = float(orientation[6])
        magnitude = math.hypot(forward_x, forward_z)
        forward_x /= magnitude
        forward_z /= magnitude
        yaw = math.atan2(-forward_z, forward_x)
        position_field.setSFVec3f(
            [
                float(position[0]) + FORWARD_OFFSET_METERS * forward_x,
                float(position[1]) + EYE_HEIGHT_METERS,
                float(position[2]) + FORWARD_OFFSET_METERS * forward_z,
            ]
        )
        orientation_field.setSFRotation(viewpoint_orientation(yaw))
        simulation_time = float(supervisor.getTime())
        if not initial_exported and simulation_time >= 1.0:
            supervisor.exportImage(str(output / "first_person_initial.png"), 100)
            initial_exported = True
        if failure.is_file():
            supervisor.exportImage(str(output / "first_person_failure.png"), 100)
            supervisor.simulationQuit(1)
            return
        if complete.is_file():
            if completion_time is None:
                completion_time = simulation_time
                supervisor.exportImage(str(output / "first_person_final.png"), 100)
            if simulation_time - completion_time >= 1.0:
                supervisor.simulationQuit(0)
                return


if __name__ == "__main__":
    main()
