from __future__ import annotations

import json
import math
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from controller import Supervisor

CONFIG_PATH = Path(__file__).resolve().parents[2] / "config" / "stage4a_motion_test.json"
LOG_PATH = Path(__file__).resolve().parents[2] / "logs" / "scenario_supervisor.jsonl"
ROBOT_DEF = "CONSTRUCTION_ROBOT"
POSITION_TOLERANCE = 0.005
ROTATION_TOLERANCE = 0.001
MAX_RESET_VERIFICATION_ATTEMPTS = 10


def append_record(record: dict[str, Any]) -> None:
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with LOG_PATH.open("a", encoding="utf-8") as file:
        file.write(json.dumps(record) + "\n")


def axis_angle_to_quaternion(
    rotation: Sequence[float],
) -> tuple[float, float, float, float]:
    if len(rotation) != 4:
        raise ValueError("An axis-angle rotation must have four values.")

    axis_x = float(rotation[0])
    axis_y = float(rotation[1])
    axis_z = float(rotation[2])
    angle = float(rotation[3])

    axis_norm = math.sqrt(axis_x * axis_x + axis_y * axis_y + axis_z * axis_z)

    if abs(angle) <= 1e-12 or axis_norm <= 1e-12:
        return (1.0, 0.0, 0.0, 0.0)

    half_angle = 0.5 * angle
    scale = math.sin(half_angle) / axis_norm

    return (
        math.cos(half_angle),
        axis_x * scale,
        axis_y * scale,
        axis_z * scale,
    )


def rotations_equivalent(
    first: Sequence[float],
    second: Sequence[float],
    *,
    tolerance: float,
) -> bool:
    first_quaternion = axis_angle_to_quaternion(first)
    second_quaternion = axis_angle_to_quaternion(second)

    dot_product = abs(
        sum(
            left * right
            for left, right in zip(
                first_quaternion,
                second_quaternion,
                strict=True,
            )
        )
    )

    return math.isclose(
        dot_product,
        1.0,
        abs_tol=tolerance,
        rel_tol=0.0,
    )


def positions_equivalent(
    first: Sequence[float],
    second: Sequence[float],
    *,
    tolerance: float,
) -> bool:
    return all(
        math.isclose(
            float(left),
            float(right),
            abs_tol=tolerance,
            rel_tol=0.0,
        )
        for left, right in zip(
            first,
            second,
            strict=True,
        )
    )


def main() -> None:
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    supervisor = Supervisor()
    time_step = int(supervisor.getBasicTimeStep())

    robot_node = supervisor.getFromDef(ROBOT_DEF)
    if robot_node is None:
        raise RuntimeError(f"Supervisor could not find DEF {ROBOT_DEF}.")

    translation_field = robot_node.getField("translation")
    rotation_field = robot_node.getField("rotation")

    if translation_field is None or rotation_field is None:
        raise RuntimeError("The robot translation or rotation field is missing.")

    initial_translation = list(translation_field.getSFVec3f())
    initial_rotation = list(rotation_field.getSFRotation())

    reset_time = float(config["supervisor_reset_time_seconds"])
    quit_time = float(config["simulation_quit_time_seconds"])

    LOG_PATH.unlink(missing_ok=True)
    append_record(
        {
            "event": "supervisor_start",
            "time_step_ms": time_step,
            "initial_translation": initial_translation,
            "initial_rotation": initial_rotation,
        }
    )

    reset_applied = False
    reset_verified = False
    verification_attempts = 0

    while supervisor.step(time_step) != -1:
        simulation_time = float(supervisor.getTime())

        if not reset_applied and simulation_time >= reset_time:
            before_translation = list(translation_field.getSFVec3f())
            before_rotation = list(rotation_field.getSFRotation())

            translation_field.setSFVec3f(initial_translation)
            rotation_field.setSFRotation(initial_rotation)
            robot_node.resetPhysics()
            reset_applied = True

            append_record(
                {
                    "event": "reset_applied",
                    "simulation_time": simulation_time,
                    "before_translation": before_translation,
                    "before_rotation": before_rotation,
                    "target_translation": initial_translation,
                    "target_rotation": initial_rotation,
                }
            )
            print(
                "STAGE4A_SUPERVISOR_RESET_APPLIED",
                flush=True,
            )
            continue

        if reset_applied and not reset_verified:
            verification_attempts += 1

            after_translation = list(translation_field.getSFVec3f())
            after_rotation = list(rotation_field.getSFRotation())

            position_ok = positions_equivalent(
                after_translation,
                initial_translation,
                tolerance=POSITION_TOLERANCE,
            )
            rotation_ok = rotations_equivalent(
                after_rotation,
                initial_rotation,
                tolerance=ROTATION_TOLERANCE,
            )
            reset_ok = position_ok and rotation_ok

            append_record(
                {
                    "event": "reset_verification",
                    "simulation_time": simulation_time,
                    "attempt": verification_attempts,
                    "after_translation": after_translation,
                    "after_rotation": after_rotation,
                    "position_ok": position_ok,
                    "rotation_ok": rotation_ok,
                    "reset_ok": reset_ok,
                }
            )

            if reset_ok:
                reset_verified = True
                print(
                    "STAGE4A_SUPERVISOR_RESET_OK",
                    flush=True,
                )
            elif verification_attempts >= MAX_RESET_VERIFICATION_ATTEMPTS:
                print(
                    "STAGE4A_SUPERVISOR_RESET_FAILED",
                    flush=True,
                )
                supervisor.simulationQuit(2)
                return

        if simulation_time >= quit_time:
            append_record(
                {
                    "event": "supervisor_complete",
                    "simulation_time": simulation_time,
                    "reset_applied": reset_applied,
                    "reset_verified": reset_verified,
                    "verification_attempts": (verification_attempts),
                }
            )
            print(
                f"STAGE4A_SUPERVISOR_COMPLETE reset_verified={reset_verified}",
                flush=True,
            )
            supervisor.simulationQuit(0 if reset_verified else 3)
            return


if __name__ == "__main__":
    main()
