from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from controller import Robot

CONFIG_PATH = Path(__file__).resolve().parents[2] / "config" / "stage4a_motion_test.json"
LOG_PATH = Path(__file__).resolve().parents[2] / "logs" / "manual_robot_test.jsonl"

LEFT_MOTOR_NAME = "left wheel motor"
RIGHT_MOTOR_NAME = "right wheel motor"
LEFT_ENCODER_NAME = "left wheel sensor"
RIGHT_ENCODER_NAME = "right wheel sensor"
PROXIMITY_SENSOR_NAMES = [f"ps{index}" for index in range(8)]


def append_record(record: dict[str, Any]) -> None:
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with LOG_PATH.open("a", encoding="utf-8") as file:
        file.write(json.dumps(record) + "\n")


def finite_or_none(value: float) -> float | None:
    numeric = float(value)
    return numeric if math.isfinite(numeric) else None


def command_for(
    segment_name: str,
    *,
    forward_velocity: float,
    turn_velocity: float,
) -> tuple[float, float]:
    if segment_name == "MOVE_FORWARD":
        return forward_velocity, forward_velocity
    if segment_name == "TURN_LEFT":
        return -turn_velocity, turn_velocity
    if segment_name == "TURN_RIGHT":
        return turn_velocity, -turn_velocity
    return 0.0, 0.0


def main() -> None:
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    robot = Robot()
    time_step = int(robot.getBasicTimeStep())

    left_motor = robot.getDevice(LEFT_MOTOR_NAME)
    right_motor = robot.getDevice(RIGHT_MOTOR_NAME)
    left_encoder = robot.getDevice(LEFT_ENCODER_NAME)
    right_encoder = robot.getDevice(RIGHT_ENCODER_NAME)
    proximity_sensors = [robot.getDevice(name) for name in PROXIMITY_SENSOR_NAMES]

    left_motor.setPosition(float("inf"))
    right_motor.setPosition(float("inf"))
    left_motor.setVelocity(0.0)
    right_motor.setVelocity(0.0)

    left_encoder.enable(time_step)
    right_encoder.enable(time_step)
    for sensor in proximity_sensors:
        sensor.enable(time_step)

    LOG_PATH.unlink(missing_ok=True)

    segments: list[dict[str, Any]] = config["segments"]
    segment_deadlines: list[float] = []
    elapsed_deadline = 0.0
    for segment in segments:
        elapsed_deadline += float(segment["duration_seconds"])
        segment_deadlines.append(elapsed_deadline)

    append_record(
        {
            "event": "run_start",
            "time_step_ms": time_step,
            "device_names": {
                "motors": [
                    LEFT_MOTOR_NAME,
                    RIGHT_MOTOR_NAME,
                ],
                "encoders": [
                    LEFT_ENCODER_NAME,
                    RIGHT_ENCODER_NAME,
                ],
                "proximity": PROXIMITY_SENSOR_NAMES,
            },
        }
    )

    current_segment_index = -1
    sample_interval = int(config["sensor_sample_interval_steps"])
    step_index = 0
    valid_sensor_samples = 0

    try:
        while robot.step(time_step) != -1:
            elapsed_seconds = float(robot.getTime())

            resolved_segment_index = len(segments) - 1
            for index, deadline in enumerate(segment_deadlines):
                if elapsed_seconds < deadline:
                    resolved_segment_index = index
                    break

            segment = segments[resolved_segment_index]
            segment_name = str(segment["name"])

            if resolved_segment_index != current_segment_index:
                current_segment_index = resolved_segment_index
                left_velocity, right_velocity = command_for(
                    segment_name,
                    forward_velocity=float(config["forward_velocity"]),
                    turn_velocity=float(config["turn_velocity"]),
                )
                left_motor.setVelocity(left_velocity)
                right_motor.setVelocity(right_velocity)
                append_record(
                    {
                        "event": "segment_start",
                        "simulation_time": elapsed_seconds,
                        "segment": segment_name,
                        "left_velocity": left_velocity,
                        "right_velocity": right_velocity,
                    }
                )
                print(f"STAGE4A_SEGMENT {segment_name} {left_velocity:.3f} {right_velocity:.3f}")

            if step_index % sample_interval == 0:
                proximity_values = [
                    finite_or_none(sensor.getValue()) for sensor in proximity_sensors
                ]
                left_encoder_value = finite_or_none(left_encoder.getValue())
                right_encoder_value = finite_or_none(right_encoder.getValue())

                sensors_valid = (
                    left_encoder_value is not None
                    and right_encoder_value is not None
                    and all(value is not None for value in proximity_values)
                )
                if sensors_valid:
                    valid_sensor_samples += 1

                append_record(
                    {
                        "event": "sensor_sample",
                        "simulation_time": elapsed_seconds,
                        "segment": segment_name,
                        "left_encoder": left_encoder_value,
                        "right_encoder": right_encoder_value,
                        "proximity": proximity_values,
                        "sensors_valid": sensors_valid,
                    }
                )

            step_index += 1
    finally:
        left_motor.setVelocity(0.0)
        right_motor.setVelocity(0.0)
        append_record(
            {
                "event": "run_complete",
                "simulation_time": float(robot.getTime()),
                "steps": step_index,
                "valid_sensor_samples": valid_sensor_samples,
                "safe_stop_applied": True,
            }
        )
        print(f"STAGE4A_ROBOT_COMPLETE valid_sensor_samples={valid_sensor_samples}")


if __name__ == "__main__":
    main()
