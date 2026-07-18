from __future__ import annotations

import json
import math
import sys
import traceback
from pathlib import Path


def find_project_root() -> Path:
    current_file = Path(__file__).resolve()

    for parent in current_file.parents:
        if (parent / "pyproject.toml").is_file():
            return parent

    raise RuntimeError("Could not locate the project root.")


PROJECT_ROOT = find_project_root()
SOURCE_ROOT = PROJECT_ROOT / "src"

if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(SOURCE_ROOT),
    )


from controller import Robot  # noqa: E402

from riskaware_saferrl.webots import (  # noqa: E402
    BridgeState,
    GridFrame,
    ObservationBridge,
    SemanticScene,
    WebotsSensorSnapshot,
)
from riskaware_saferrl.webots.live_bridge import (  # noqa: E402
    LiveBridgeTelemetry,
)

LEFT_MOTOR_NAME = "left wheel motor"
RIGHT_MOTOR_NAME = "right wheel motor"

GPS_NAME = "gps"
INERTIAL_UNIT_NAME = "inertial unit"
COMPASS_NAME = "compass"

EMITTER_NAME = "bridge emitter"
RECEIVER_NAME = "bridge receiver"

PROXIMITY_SENSOR_NAMES = tuple(f"ps{index}" for index in range(8))

EMISSION_INTERVAL_STEPS = 16

ROBOT_LOG_PATH = PROJECT_ROOT / "webots" / "logs" / "stage4b2_live_robot.jsonl"


def append_jsonl(
    payload: dict[str, object],
) -> None:
    ROBOT_LOG_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with ROBOT_LOG_PATH.open(
        "a",
        encoding="utf-8",
        newline="\n",
    ) as log_file:
        log_file.write(
            json.dumps(
                payload,
                sort_keys=True,
            )
            + "\n"
        )


def finite_vector(
    values: list[float],
) -> bool:
    return all(math.isfinite(float(value)) for value in values)


def resolve_wheel_velocities(
    simulation_time: float,
) -> tuple[str, float, float]:
    cycle_time = simulation_time % 20.0

    if cycle_time < 1.0:
        return (
            "STOP_INITIAL",
            0.0,
            0.0,
        )

    if cycle_time < 7.0:
        return (
            "MOVE_FORWARD_1",
            4.0,
            4.0,
        )

    if cycle_time < 9.0:
        return (
            "TURN_LEFT",
            -1.8,
            1.8,
        )

    if cycle_time < 15.0:
        return (
            "MOVE_FORWARD_2",
            4.0,
            4.0,
        )

    if cycle_time < 17.0:
        return (
            "TURN_RIGHT",
            1.8,
            -1.8,
        )

    return (
        "STOP_FINAL",
        0.0,
        0.0,
    )


def main() -> None:
    robot = Robot()
    time_step = int(robot.getBasicTimeStep())

    left_motor = robot.getDevice(LEFT_MOTOR_NAME)
    right_motor = robot.getDevice(RIGHT_MOTOR_NAME)

    left_motor.setPosition(float("inf"))
    right_motor.setPosition(float("inf"))

    left_motor.setVelocity(0.0)
    right_motor.setVelocity(0.0)

    gps = robot.getDevice(GPS_NAME)
    inertial_unit = robot.getDevice(INERTIAL_UNIT_NAME)
    compass = robot.getDevice(COMPASS_NAME)

    emitter = robot.getDevice(EMITTER_NAME)
    receiver = robot.getDevice(RECEIVER_NAME)

    proximity_sensors = tuple(robot.getDevice(name) for name in PROXIMITY_SENSOR_NAMES)

    gps.enable(time_step)
    inertial_unit.enable(time_step)
    compass.enable(time_step)
    receiver.enable(time_step)

    for sensor in proximity_sensors:
        sensor.enable(time_step)

    frame = GridFrame(
        size=12,
        x_min=-5.0,
        x_max=5.0,
        z_min=-4.0,
        z_max=4.0,
    )

    scene = SemanticScene(
        size=12,
        obstacles=frozenset(
            {
                (5, 6),
                (7, 5),
            }
        ),
        hazards=frozenset(
            {
                (6, 8),
                (4, 7),
            }
        ),
        workers=frozenset(
            {
                (7, 7),
            }
        ),
        restricted=frozenset(
            {
                (6, 4),
            }
        ),
    )

    observation_bridge = ObservationBridge(
        vision_radius=4,
        inspection_radius=2,
    )

    visited: set[tuple[int, int]] = set()
    inspected: set[tuple[int, int]] = set()

    robot_step_count = 0
    sample_index = 0
    ack_count = 0
    last_motion_name = ""

    print(
        "STAGE4B2_ROBOT_READY",
        flush=True,
    )

    try:
        while robot.step(time_step) != -1:
            robot_step_count += 1

            while receiver.getQueueLength() > 0:
                try:
                    acknowledgement = json.loads(receiver.getString())

                    if acknowledgement.get("type") == "stage4b2_ack":
                        ack_count += 1

                        print(
                            f"STAGE4B2_ACK {ack_count}",
                            flush=True,
                        )
                except (
                    json.JSONDecodeError,
                    AttributeError,
                ):
                    pass

                receiver.nextPacket()

            simulation_time = float(robot.getTime())

            (
                motion_name,
                left_velocity,
                right_velocity,
            ) = resolve_wheel_velocities(simulation_time)

            left_motor.setVelocity(left_velocity)
            right_motor.setVelocity(right_velocity)

            if motion_name != last_motion_name:
                last_motion_name = motion_name

                print(
                    f"STAGE4B2_MOTION {motion_name} {left_velocity:.2f} {right_velocity:.2f}",
                    flush=True,
                )

            if robot_step_count % EMISSION_INTERVAL_STEPS != 0:
                continue

            gps_values = list(gps.getValues())

            roll_pitch_yaw = list(inertial_unit.getRollPitchYaw())

            compass_values = list(compass.getValues())

            proximity_values = [float(sensor.getValue()) for sensor in proximity_sensors]

            if not (
                finite_vector(gps_values)
                and finite_vector(roll_pitch_yaw)
                and finite_vector(compass_values)
                and finite_vector(proximity_values)
            ):
                continue

            snapshot = WebotsSensorSnapshot(
                x=float(gps_values[0]),
                z=float(gps_values[2]),
                yaw_radians=float(roll_pitch_yaw[2]),
                proximity=tuple(proximity_values),
                step_count=sample_index,
                max_steps=250,
                inspected_ratio=(
                    len(inspected)
                    / max(
                        1,
                        len(scene.hazards),
                    )
                ),
            )

            try:
                agent_position = snapshot.grid_position(frame)
            except ValueError:
                left_motor.setVelocity(0.0)
                right_motor.setVelocity(0.0)

                print(
                    "STAGE4B2_SAFE_STOP reason=outside_grid",
                    flush=True,
                )

                continue

            visited.add(agent_position)

            bridge_state = BridgeState(
                agent_position=agent_position,
                visited=frozenset(visited),
                inspected=frozenset(inspected),
                steps=sample_index,
                max_steps=250,
            )

            observation = observation_bridge.build_observation(
                scene,
                bridge_state,
            )

            action_mask = observation_bridge.action_mask(
                scene,
                bridge_state,
            )

            telemetry = LiveBridgeTelemetry.from_observation(
                sample_index=sample_index,
                simulation_time=(simulation_time),
                snapshot=snapshot,
                frame=frame,
                compass=tuple(float(value) for value in compass_values),
                observation=observation,
                action_mask=action_mask,
                ack_count=ack_count,
            )

            encoded = telemetry.to_json()

            emitter.send(encoded.encode("utf-8"))

            append_jsonl(telemetry.to_dict())

            print(
                "STAGE4B2_SAMPLE "
                f"index={sample_index} "
                f"grid={agent_position} "
                f"heading={telemetry.heading} "
                f"valid_actions="
                f"{sum(action_mask)}",
                flush=True,
            )

            sample_index += 1
    finally:
        left_motor.setVelocity(0.0)
        right_motor.setVelocity(0.0)

        print(
            "STAGE4B2_ROBOT_SAFE_STOP",
            flush=True,
        )


if __name__ == "__main__":
    startup_path = PROJECT_ROOT / "webots" / "logs" / "stage4b2_robot_startup.txt"

    error_path = PROJECT_ROOT / "webots" / "logs" / "stage4b2_robot_error.log"

    startup_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    startup_path.write_text(
        "controller entered main\n",
        encoding="utf-8",
    )

    try:
        main()
    except Exception:
        error_path.write_text(
            traceback.format_exc(),
            encoding="utf-8",
        )
        raise
