from __future__ import annotations

import json
from pathlib import Path

from controller import Robot

LOG_PATH = Path(__file__).resolve().parents[2] / "logs" / "construction_robot_safe_stop.jsonl"


def main() -> None:
    robot = Robot()
    time_step = int(robot.getBasicTimeStep())

    left_motor = robot.getDevice("left wheel motor")
    right_motor = robot.getDevice("right wheel motor")

    left_motor.setPosition(float("inf"))
    right_motor.setPosition(float("inf"))
    left_motor.setVelocity(0.0)
    right_motor.setVelocity(0.0)

    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    LOG_PATH.write_text(
        json.dumps(
            {
                "event": "safe_stop_controller_start",
                "time_step_ms": time_step,
                "policy_connected": False,
                "shield_connected": False,
                "motors_stopped": True,
            }
        )
        + "\n",
        encoding="utf-8",
    )

    print("STAGE4A_SAFE_STOP_CONTROLLER_READY")

    while robot.step(time_step) != -1:
        left_motor.setVelocity(0.0)
        right_motor.setVelocity(0.0)


if __name__ == "__main__":
    main()
