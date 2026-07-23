from __future__ import annotations

import hashlib
import json
import math
import os
import re
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from controller import Node, Robot

CAMERA_NAME = "inspection camera"
GPS_NAME = "gps"
INERTIAL_UNIT_NAME = "inertial unit"
COMPASS_NAME = "compass"
LEFT_MOTOR_NAME = "left wheel motor"
RIGHT_MOTOR_NAME = "right wheel motor"
LEFT_WHEEL_SENSOR_NAME = "left wheel sensor"
RIGHT_WHEEL_SENSOR_NAME = "right wheel sensor"

FORWARD_CALIBRATION_SECONDS = 3.5
TURN_CALIBRATION_SECONDS = 5.0
CALIBRATION_MINIMUM_DISPLACEMENT = 0.10
CALIBRATION_MINIMUM_YAW_CHANGE = math.radians(8.0)

YAW_TO_WORLD_SIGN = -1.0
FORWARD_MOTOR_SIGN = -1.0

NAVIGATION_HEADING_GAIN = 0.45
NAVIGATION_DISTANCE_GAIN = 0.75
ALIGNMENT_HEADING_LIMIT_RADIANS = math.radians(5.0)
SCAN_HEADING_GAIN = 0.55

MAXIMUM_RECOVERY_COUNT = 3
STAGNATION_TIMEOUT_SECONDS = 12.0
MINIMUM_PROGRESS_METERS = 0.02

MAXIMUM_SAFE_ROLL_RADIANS = math.radians(10.0)
MAXIMUM_SAFE_PITCH_RADIANS = math.radians(10.0)
TILT_GUARD_GRACE_SECONDS = 2.5
MAXIMUM_HARD_TILT_RADIANS = math.radians(18.0)
TILT_FAILURE_DURATION_SECONDS = 0.75
MOTOR_COMMAND_SLEW_RATE_RADIANS_PER_SECOND = 0.50
RECOVERY_DURATION_SECONDS = 1.1
PROXIMITY_TRIGGER_CONSECUTIVE_STEPS = 4
PROXIMITY_TRIGGER_MINIMUM_RATIO = 0.985

STATE_CALIBRATE_FORWARD = "CALIBRATE_FORWARD"
STATE_CALIBRATE_FORWARD_SETTLE = "CALIBRATE_FORWARD_SETTLE"
STATE_CALIBRATE_TURN = "CALIBRATE_TURN"
STATE_CALIBRATE_TURN_SETTLE = "CALIBRATE_TURN_SETTLE"
STATE_NAVIGATE = "NAVIGATE"
STATE_DWELL = "DWELL"
STATE_SCAN = "SCAN"
STATE_RECOVERY = "RECOVERY"
STATE_COMPLETE = "COMPLETE"


@dataclass(frozen=True)
class Waypoint:
    name: str
    x: float
    z: float
    scan: bool
    evidence: bool


@dataclass
class DistanceSensorState:
    name: str
    device: Any
    minimum: float
    maximum: float
    baseline_ratio: float = 0.0


def normalize_angle(
    angle: float,
) -> float:
    return math.atan2(
        math.sin(angle),
        math.cos(angle),
    )


def clamp(
    value: float,
    minimum: float,
    maximum: float,
) -> float:
    return max(
        minimum,
        min(maximum, value),
    )


def move_toward(
    current: float,
    target: float,
    maximum_delta: float,
) -> float:
    if target > current:
        return min(target, current + maximum_delta)

    return max(target, current - maximum_delta)


def y_up_attitude_from_quaternion(
    quaternion: list[float],
) -> tuple[float, float]:
    """Return X/Z tilt angles from a Webots [x, y, z, w] quaternion."""
    x, y, z, w = finite_list(quaternion, name="orientation_quaternion")
    norm = math.sqrt(x * x + y * y + z * z + w * w)

    if norm <= 1e-12:
        raise RuntimeError("The orientation quaternion has zero norm.")

    x, y, z, w = (value / norm for value in (x, y, z, w))
    up_x = 2.0 * (x * y - z * w)
    up_y = 1.0 - 2.0 * (x * x + z * z)
    up_z = 2.0 * (y * z + x * w)

    roll_x = math.atan2(up_z, up_y)
    pitch_z = math.atan2(-up_x, up_y)
    return roll_x, pitch_z


def project_root() -> Path:
    value = os.environ.get("RISK_AWARE_PROJECT_ROOT")

    if not value:
        raise RuntimeError("RISK_AWARE_PROJECT_ROOT is not set.")

    return Path(value).resolve()


def route_config_path() -> Path:
    return project_root() / "configs" / "webots" / "stage5a3_closed_loop_route.json"


def output_directory() -> Path:
    return project_root() / "webots" / "logs" / "stage5a3_closed_loop"


def load_route_config() -> dict[str, Any]:
    path = route_config_path()

    if not path.is_file():
        raise RuntimeError(f"Route configuration is missing: {path}")

    return json.loads(path.read_text(encoding="utf-8"))


def parse_waypoints(
    route_config: dict[str, Any],
) -> list[Waypoint]:
    waypoints = [
        Waypoint(
            name=str(item["name"]),
            x=float(item["x"]),
            z=float(item["z"]),
            scan=bool(item["scan"]),
            evidence=bool(item["evidence"]),
        )
        for item in route_config["waypoints"]
    ]

    if len(waypoints) < 7:
        raise RuntimeError("At least seven waypoints are required.")

    return waypoints


def finite_list(
    values: list[float],
    *,
    name: str,
) -> list[float]:
    converted = [float(value) for value in values]

    if not all(math.isfinite(value) for value in converted):
        raise RuntimeError(f"{name} contains a non-finite value.")

    return converted


def write_json(
    path: Path,
    payload: dict[str, Any],
) -> None:
    temporary_path = path.with_suffix(path.suffix + ".tmp")

    temporary_path.write_text(
        json.dumps(
            payload,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
        newline="\n",
    )

    temporary_path.replace(path)


def image_entropy(
    grayscale: np.ndarray,
) -> float:
    histogram = cv2.calcHist(
        [grayscale],
        [0],
        None,
        [256],
        [0, 256],
    ).ravel()

    total = float(histogram.sum())

    if total <= 0.0:
        return 0.0

    probabilities = histogram / total
    probabilities = probabilities[probabilities > 0.0]

    return float(-np.sum(probabilities * np.log2(probabilities)))


def safe_filename(
    value: str,
) -> str:
    normalized = re.sub(
        r"[^A-Za-z0-9_-]+",
        "_",
        value,
    )

    return normalized.strip("_").lower()


def read_camera_frame(
    camera: Any,
    *,
    width: int,
    height: int,
) -> tuple[bytes, np.ndarray, np.ndarray]:
    raw_image = camera.getImage()

    if raw_image is None:
        raise RuntimeError("Camera returned no image.")

    image_bytes = bytes(raw_image)
    expected_byte_length = width * height * 4

    if len(image_bytes) != expected_byte_length:
        raise RuntimeError(
            "Unexpected camera byte length. "
            f"Expected {expected_byte_length}, "
            f"received {len(image_bytes)}."
        )

    frame_bgra = np.frombuffer(
        image_bytes,
        dtype=np.uint8,
    ).reshape(
        height,
        width,
        4,
    )

    frame_bgr = frame_bgra[:, :, :3].copy()

    grayscale = cv2.cvtColor(
        frame_bgr,
        cv2.COLOR_BGR2GRAY,
    )

    return image_bytes, frame_bgr, grayscale


def discover_distance_sensors(
    robot: Robot,
    *,
    time_step: int,
) -> list[DistanceSensorState]:
    sensors: list[DistanceSensorState] = []

    for index in range(robot.getNumberOfDevices()):
        device = robot.getDeviceByIndex(index)

        if device.getNodeType() != Node.DISTANCE_SENSOR:
            continue

        name = str(device.getName())

        device.enable(time_step)

        minimum = float(device.getMinValue())

        maximum = float(device.getMaxValue())

        sensors.append(
            DistanceSensorState(
                name=name,
                device=device,
                minimum=minimum,
                maximum=maximum,
            )
        )

    return sensors


def normalized_sensor_ratio(
    sensor: DistanceSensorState,
) -> float:
    value = float(sensor.device.getValue())

    span = sensor.maximum - sensor.minimum

    if not math.isfinite(value) or span <= 0.0:
        return 0.0

    return clamp(
        (value - sensor.minimum) / span,
        0.0,
        1.0,
    )


def maximum_proximity_ratio(
    sensors: list[DistanceSensorState],
) -> float:
    if not sensors:
        return 0.0

    return max(normalized_sensor_ratio(sensor) for sensor in sensors)


def save_evidence_frame(
    *,
    evidence_directory: Path,
    waypoint_name: str,
    view_name: str,
    frame_bgr: np.ndarray,
    evidence_paths: list[str],
) -> str:
    filename = f"{safe_filename(waypoint_name)}_{safe_filename(view_name)}.png"

    path = evidence_directory / filename

    saved = cv2.imwrite(
        str(path),
        frame_bgr,
    )

    if not saved:
        raise RuntimeError(f"Could not save evidence frame: {path}")

    path_string = str(path)

    if path_string not in evidence_paths:
        evidence_paths.append(path_string)

    return path_string


def main() -> None:
    route_config = load_route_config()
    waypoints = parse_waypoints(route_config)

    arrival_tolerance = float(route_config["arrival_tolerance_meters"])

    heading_tolerance = math.radians(float(route_config["heading_tolerance_degrees"]))

    scan_angle = math.radians(float(route_config["scan_angle_degrees"]))

    minimum_capture_count = int(route_config["minimum_capture_count"])

    capture_interval_seconds = float(route_config["capture_interval_seconds"])

    maximum_simulation_time_seconds = float(route_config["maximum_simulation_time_seconds"])

    root = output_directory()
    evidence_directory = root / "evidence_frames"

    root.mkdir(
        parents=True,
        exist_ok=True,
    )

    evidence_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    telemetry_path = root / "stage5a3_mission_telemetry.jsonl"

    summary_path = root / "stage5a3_mission_summary.json"

    completion_marker_path = root / "stage5a3_complete.marker"

    robot = Robot()
    time_step = int(robot.getBasicTimeStep())

    camera = robot.getDevice(CAMERA_NAME)
    gps = robot.getDevice(GPS_NAME)
    inertial_unit = robot.getDevice(INERTIAL_UNIT_NAME)
    compass = robot.getDevice(COMPASS_NAME)
    left_motor = robot.getDevice(LEFT_MOTOR_NAME)
    right_motor = robot.getDevice(RIGHT_MOTOR_NAME)
    left_wheel_sensor = robot.getDevice(LEFT_WHEEL_SENSOR_NAME)
    right_wheel_sensor = robot.getDevice(RIGHT_WHEEL_SENSOR_NAME)

    devices = {
        CAMERA_NAME: camera,
        GPS_NAME: gps,
        INERTIAL_UNIT_NAME: inertial_unit,
        COMPASS_NAME: compass,
        LEFT_MOTOR_NAME: left_motor,
        RIGHT_MOTOR_NAME: right_motor,
        LEFT_WHEEL_SENSOR_NAME: left_wheel_sensor,
        RIGHT_WHEEL_SENSOR_NAME: right_wheel_sensor,
    }

    for name, device in devices.items():
        if device is None:
            raise RuntimeError(f"Webots device was not found: {name}")

    camera.enable(time_step)
    gps.enable(time_step)
    inertial_unit.enable(time_step)
    compass.enable(time_step)
    left_wheel_sensor.enable(time_step)
    right_wheel_sensor.enable(time_step)

    distance_sensors = discover_distance_sensors(
        robot,
        time_step=time_step,
    )

    left_motor.setPosition(float("inf"))
    right_motor.setPosition(float("inf"))
    left_motor.setVelocity(0.0)
    right_motor.setVelocity(0.0)

    left_maximum_velocity = float(left_motor.getMaxVelocity())
    right_maximum_velocity = float(right_motor.getMaxVelocity())

    maximum_motor_velocity = min(
        left_maximum_velocity,
        right_maximum_velocity,
    )

    if not math.isfinite(maximum_motor_velocity) or maximum_motor_velocity <= 0.0:
        maximum_motor_velocity = 8.0

    cruise_velocity = 0.20 * maximum_motor_velocity
    minimum_forward_velocity = 0.06 * maximum_motor_velocity
    turn_velocity_limit = 0.025 * maximum_motor_velocity
    calibration_forward_velocity = 0.10 * maximum_motor_velocity
    calibration_turn_velocity = 0.025 * maximum_motor_velocity
    recovery_turn_velocity = 0.025 * maximum_motor_velocity

    width = int(camera.getWidth())
    height = int(camera.getHeight())

    if (width, height) != (640, 360):
        raise RuntimeError(f"Unexpected camera resolution: {width}x{height}")

    expected_byte_length = width * height * 4

    state = STATE_CALIBRATE_FORWARD
    state_started_at = 0.0

    calibration_start_position: np.ndarray | None = None
    calibration_start_yaw: float | None = None
    calibration_displacement = 0.0
    heading_offset = 0.0
    turn_command_sign = 1.0

    target_waypoint_index = 1
    visited_waypoint_names = [waypoints[0].name]
    arrival_records: list[dict[str, Any]] = []
    evidence_paths: list[str] = []

    initial_position: np.ndarray | None = None
    last_step_position: np.ndarray | None = None
    last_capture_grayscale: np.ndarray | None = None

    path_length = 0.0
    total_heading_change = 0.0
    previous_world_heading: float | None = None

    capture_records: list[dict[str, Any]] = []
    unique_checksums: set[str] = set()
    next_capture_time = 0.0

    arrival_heading = 0.0
    scan_targets: list[tuple[str, float]] = []
    active_scan_view = ""
    dwell_until = 0.0

    best_distance_to_target = float("inf")
    last_progress_time = 0.0
    recovery_until = 0.0
    recovery_count = 0
    obstacle_interventions = 0
    proximity_trigger_count = 0
    tilt_limit_started_at: float | None = None
    applied_left_command = 0.0
    applied_right_command = 0.0

    route_completed = False
    route_completed_at = 0.0

    print(
        "STAGE5A3_CONTROLLER_READY "
        f"resolution={width}x{height} "
        f"expected_bytes={expected_byte_length} "
        f"waypoints={len(waypoints)} "
        f"distance_sensors={len(distance_sensors)}",
        flush=True,
    )

    with telemetry_path.open(
        "w",
        encoding="utf-8",
        newline="\n",
    ) as telemetry_file:
        while robot.step(time_step) != -1:
            simulation_time = float(robot.getTime())

            if simulation_time > maximum_simulation_time_seconds:
                raise RuntimeError(
                    "The closed-loop mission exceeded the configured simulation timeout."
                )

            gps_values = finite_list(
                list(gps.getValues()),
                name="gps",
            )

            rpy_values = finite_list(
                list(inertial_unit.getRollPitchYaw()),
                name="inertial_unit",
            )

            orientation_quaternion = finite_list(
                list(inertial_unit.getQuaternion()),
                name="orientation_quaternion",
            )

            compass_values = finite_list(
                list(compass.getValues()),
                name="compass",
            )

            current_position = np.asarray(
                [gps_values[0], gps_values[2]],
                dtype=np.float64,
            )

            current_roll, current_pitch = y_up_attitude_from_quaternion(orientation_quaternion)
            compass_heading = math.atan2(float(compass_values[0]), float(compass_values[2]))
            current_yaw = -compass_heading
            left_wheel_position = float(left_wheel_sensor.getValue())
            right_wheel_position = float(right_wheel_sensor.getValue())

            if simulation_time >= TILT_GUARD_GRACE_SECONDS:
                tilt_magnitude = max(
                    abs(current_roll),
                    abs(current_pitch),
                )

                if tilt_magnitude >= MAXIMUM_HARD_TILT_RADIANS:
                    left_motor.setVelocity(0.0)
                    right_motor.setVelocity(0.0)

                    raise RuntimeError(
                        "Robot tilt exceeded the hard safety limit. "
                        f"roll={math.degrees(current_roll):.3f} "
                        f"pitch={math.degrees(current_pitch):.3f}"
                    )

                soft_tilt_exceeded = (
                    abs(current_roll) > MAXIMUM_SAFE_ROLL_RADIANS
                    or abs(current_pitch) > MAXIMUM_SAFE_PITCH_RADIANS
                )

                if soft_tilt_exceeded:
                    if tilt_limit_started_at is None:
                        tilt_limit_started_at = simulation_time
                    elif simulation_time - tilt_limit_started_at >= TILT_FAILURE_DURATION_SECONDS:
                        left_motor.setVelocity(0.0)
                        right_motor.setVelocity(0.0)

                        raise RuntimeError(
                            "Robot tilt remained above the validated limit. "
                            f"roll={math.degrees(current_roll):.3f} "
                            f"pitch={math.degrees(current_pitch):.3f}"
                        )
                else:
                    tilt_limit_started_at = None

            if initial_position is None:
                initial_position = current_position.copy()

            if last_step_position is not None:
                path_length += float(np.linalg.norm(current_position - last_step_position))

            last_step_position = current_position.copy()

            current_world_heading = normalize_angle(
                YAW_TO_WORLD_SIGN * current_yaw + heading_offset
            )

            if previous_world_heading is not None:
                total_heading_change += abs(
                    normalize_angle(current_world_heading - previous_world_heading)
                )

            previous_world_heading = current_world_heading

            proximity_ratio = maximum_proximity_ratio(distance_sensors)

            if state == STATE_CALIBRATE_FORWARD:
                if calibration_start_position is None:
                    calibration_start_position = current_position.copy()
                    calibration_start_yaw = current_yaw
                    state_started_at = simulation_time

                left_command = FORWARD_MOTOR_SIGN * calibration_forward_velocity
                right_command = FORWARD_MOTOR_SIGN * calibration_forward_velocity

                elapsed = simulation_time - state_started_at

                if elapsed >= FORWARD_CALIBRATION_SECONDS:
                    displacement_vector = current_position - calibration_start_position

                    calibration_displacement = float(np.linalg.norm(displacement_vector))

                    if calibration_displacement < CALIBRATION_MINIMUM_DISPLACEMENT:
                        raise RuntimeError(
                            "Forward calibration displacement "
                            "was too small: "
                            f"{calibration_displacement:.4f} meters."
                        )

                    measured_world_heading = math.atan2(
                        float(displacement_vector[1]),
                        float(displacement_vector[0]),
                    )

                    heading_offset = normalize_angle(
                        measured_world_heading - YAW_TO_WORLD_SIGN * float(calibration_start_yaw)
                    )

                    state = STATE_CALIBRATE_FORWARD_SETTLE
                    state_started_at = simulation_time

                    print(
                        "STAGE5A3_FORWARD_CALIBRATION_COMPLETE "
                        f"displacement={calibration_displacement:.4f} "
                        f"heading_offset_degrees="
                        f"{math.degrees(heading_offset):.3f}",
                        flush=True,
                    )

            elif state == STATE_CALIBRATE_FORWARD_SETTLE:
                left_command = 0.0
                right_command = 0.0

                if simulation_time - state_started_at >= 2.0:
                    state = STATE_CALIBRATE_TURN
                    state_started_at = simulation_time
                    calibration_start_yaw = current_yaw

            elif state == STATE_CALIBRATE_TURN:
                left_command = -calibration_turn_velocity
                right_command = calibration_turn_velocity

                elapsed = simulation_time - state_started_at
                yaw_change = normalize_angle(current_yaw - float(calibration_start_yaw))

                if abs(yaw_change) >= CALIBRATION_MINIMUM_YAW_CHANGE:
                    turn_command_sign = YAW_TO_WORLD_SIGN * (1.0 if yaw_change > 0.0 else -1.0)

                    state = STATE_CALIBRATE_TURN_SETTLE
                    state_started_at = simulation_time

                    print(
                        "STAGE5A3_TURN_CALIBRATION_COMPLETE "
                        f"yaw_change_degrees="
                        f"{math.degrees(yaw_change):.3f} "
                        f"turn_command_sign={turn_command_sign:.1f} "
                        f"yaw_to_world_sign={YAW_TO_WORLD_SIGN:.1f} "
                        f"forward_motor_sign={FORWARD_MOTOR_SIGN:.1f}",
                        flush=True,
                    )
                elif elapsed >= TURN_CALIBRATION_SECONDS:
                    raise RuntimeError(
                        f"Turn calibration yaw change was too small: {yaw_change:.5f} radians."
                    )

            elif state == STATE_CALIBRATE_TURN_SETTLE:
                left_command = 0.0
                right_command = 0.0

                if simulation_time - state_started_at >= 2.0:
                    state = STATE_NAVIGATE
                    state_started_at = simulation_time
                    best_distance_to_target = float("inf")
                    last_progress_time = simulation_time

            elif state == STATE_NAVIGATE:
                target = waypoints[target_waypoint_index]

                delta_x = target.x - float(current_position[0])
                delta_z = target.z - float(current_position[1])

                distance_to_target = math.hypot(
                    delta_x,
                    delta_z,
                )

                target_heading = math.atan2(
                    delta_z,
                    delta_x,
                )

                heading_error = normalize_angle(target_heading - current_world_heading)

                if distance_to_target < best_distance_to_target - MINIMUM_PROGRESS_METERS:
                    best_distance_to_target = distance_to_target
                    last_progress_time = simulation_time

                if proximity_ratio >= PROXIMITY_TRIGGER_MINIMUM_RATIO:
                    proximity_trigger_count += 1
                else:
                    proximity_trigger_count = 0

                obstacle_triggered = proximity_trigger_count >= PROXIMITY_TRIGGER_CONSECUTIVE_STEPS

                heading_aligned_for_progress = abs(heading_error) <= ALIGNMENT_HEADING_LIMIT_RADIANS

                if not heading_aligned_for_progress:
                    last_progress_time = simulation_time

                stagnated = (
                    heading_aligned_for_progress
                    and simulation_time - last_progress_time >= STAGNATION_TIMEOUT_SECONDS
                )

                if obstacle_triggered or stagnated:
                    recovery_count += 1

                    if recovery_count > MAXIMUM_RECOVERY_COUNT:
                        raise RuntimeError(
                            "The waypoint navigator exceeded the maximum recovery count."
                        )

                    if obstacle_triggered:
                        obstacle_interventions += 1

                    state = STATE_RECOVERY
                    recovery_until = simulation_time + RECOVERY_DURATION_SECONDS
                    proximity_trigger_count = 0

                    direction = 1.0 if recovery_count % 2 == 1 else -1.0

                    left_command = -direction * recovery_turn_velocity
                    right_command = direction * recovery_turn_velocity

                    print(
                        "STAGE5A3_RECOVERY_STARTED "
                        f"count={recovery_count} "
                        f"reason="
                        f"{'obstacle' if obstacle_triggered else 'stagnation'} "
                        f"waypoint={target.name}",
                        flush=True,
                    )

                elif distance_to_target <= arrival_tolerance:
                    left_command = 0.0
                    right_command = 0.0

                    arrival_record = {
                        "waypoint_index": (target_waypoint_index),
                        "waypoint_name": target.name,
                        "simulation_time_seconds": (simulation_time),
                        "arrival_distance_meters": (distance_to_target),
                        "position_x": float(current_position[0]),
                        "position_z": float(current_position[1]),
                        "world_heading_degrees": (math.degrees(current_world_heading)),
                    }

                    arrival_records.append(arrival_record)
                    visited_waypoint_names.append(target.name)

                    image_bytes, frame_bgr, _ = read_camera_frame(
                        camera,
                        width=width,
                        height=height,
                    )

                    del image_bytes

                    if target.evidence:
                        save_evidence_frame(
                            evidence_directory=(evidence_directory),
                            waypoint_name=target.name,
                            view_name="center",
                            frame_bgr=frame_bgr,
                            evidence_paths=evidence_paths,
                        )

                    print(
                        "STAGE5A3_WAYPOINT_REACHED "
                        f"index={target_waypoint_index} "
                        f"name={target.name} "
                        f"distance={distance_to_target:.3f}",
                        flush=True,
                    )

                    arrival_heading = current_world_heading

                    if target.scan:
                        scan_targets = [
                            (
                                "left",
                                normalize_angle(arrival_heading + scan_angle),
                            ),
                            (
                                "right",
                                normalize_angle(arrival_heading - scan_angle),
                            ),
                            (
                                "recenter",
                                arrival_heading,
                            ),
                        ]
                        active_scan_view = ""
                        state = STATE_SCAN
                        state_started_at = simulation_time
                    else:
                        state = STATE_DWELL
                        dwell_until = simulation_time + 0.55

                else:
                    absolute_heading_error = abs(heading_error)

                    if absolute_heading_error > ALIGNMENT_HEADING_LIMIT_RADIANS:
                        forward_command = 0.0
                    else:
                        alignment_scale = max(
                            0.20,
                            math.cos(absolute_heading_error),
                        )

                        distance_velocity = clamp(
                            NAVIGATION_DISTANCE_GAIN * distance_to_target,
                            minimum_forward_velocity,
                            cruise_velocity,
                        )

                        forward_command = FORWARD_MOTOR_SIGN * distance_velocity * alignment_scale

                        if distance_to_target < 0.80:
                            forward_command *= 0.55

                    turn_command = clamp(
                        turn_command_sign
                        * NAVIGATION_HEADING_GAIN
                        * heading_error
                        * maximum_motor_velocity,
                        -turn_velocity_limit,
                        turn_velocity_limit,
                    )

                    left_command = forward_command - turn_command
                    right_command = forward_command + turn_command

            elif state == STATE_RECOVERY:
                direction = 1.0 if recovery_count % 2 == 1 else -1.0

                left_command = -direction * recovery_turn_velocity
                right_command = direction * recovery_turn_velocity

                if simulation_time >= recovery_until:
                    state = STATE_NAVIGATE
                    best_distance_to_target = float("inf")
                    last_progress_time = simulation_time

                    print(
                        f"STAGE5A3_RECOVERY_COMPLETE count={recovery_count}",
                        flush=True,
                    )

            elif state == STATE_SCAN:
                target = waypoints[target_waypoint_index]

                if not scan_targets:
                    state = STATE_DWELL
                    dwell_until = simulation_time + 0.45
                    left_command = 0.0
                    right_command = 0.0
                else:
                    view_name, scan_heading = scan_targets[0]
                    active_scan_view = view_name

                    scan_error = normalize_angle(scan_heading - current_world_heading)

                    if abs(scan_error) <= heading_tolerance:
                        left_command = 0.0
                        right_command = 0.0

                        _, frame_bgr, _ = read_camera_frame(
                            camera,
                            width=width,
                            height=height,
                        )

                        save_evidence_frame(
                            evidence_directory=(evidence_directory),
                            waypoint_name=target.name,
                            view_name=view_name,
                            frame_bgr=frame_bgr,
                            evidence_paths=evidence_paths,
                        )

                        scan_targets.pop(0)

                        print(
                            f"STAGE5A3_SCAN_VIEW_CAPTURED waypoint={target.name} view={view_name}",
                            flush=True,
                        )
                    else:
                        scan_turn = clamp(
                            turn_command_sign
                            * SCAN_HEADING_GAIN
                            * scan_error
                            * maximum_motor_velocity,
                            -turn_velocity_limit,
                            turn_velocity_limit,
                        )

                        left_command = -scan_turn
                        right_command = scan_turn

            elif state == STATE_DWELL:
                left_command = 0.0
                right_command = 0.0

                if simulation_time >= dwell_until:
                    if target_waypoint_index >= len(waypoints) - 1:
                        state = STATE_COMPLETE
                        route_completed = True
                        route_completed_at = simulation_time

                        print(
                            "STAGE5A3_ROUTE_COMPLETE "
                            f"waypoints={len(visited_waypoint_names)}/"
                            f"{len(waypoints)} "
                            f"path_length={path_length:.3f}",
                            flush=True,
                        )
                    else:
                        target_waypoint_index += 1
                        state = STATE_NAVIGATE
                        best_distance_to_target = float("inf")
                        last_progress_time = simulation_time

            elif state == STATE_COMPLETE:
                left_command = 0.0
                right_command = 0.0

            else:
                raise RuntimeError(f"Unknown mission state: {state}")

            left_command = clamp(
                float(left_command),
                -maximum_motor_velocity,
                maximum_motor_velocity,
            )

            right_command = clamp(
                float(right_command),
                -maximum_motor_velocity,
                maximum_motor_velocity,
            )

            maximum_command_delta = MOTOR_COMMAND_SLEW_RATE_RADIANS_PER_SECOND * time_step / 1000.0

            applied_left_command = move_toward(
                applied_left_command,
                left_command,
                maximum_command_delta,
            )
            applied_right_command = move_toward(
                applied_right_command,
                right_command,
                maximum_command_delta,
            )

            left_command = applied_left_command
            right_command = applied_right_command

            left_motor.setVelocity(left_command)
            right_motor.setVelocity(right_command)

            if simulation_time >= next_capture_time:
                image_bytes, frame_bgr, grayscale = read_camera_frame(
                    camera,
                    width=width,
                    height=height,
                )

                if not capture_records:
                    preview_path = output_directory() / "live_camera_preview.png"

                    if not cv2.imwrite(str(preview_path), frame_bgr):
                        raise RuntimeError("Could not save the live camera preview.")

                    print(
                        f"STAGE5A3_CAMERA_PREVIEW path={preview_path}",
                        flush=True,
                    )

                    save_evidence_frame(
                        evidence_directory=evidence_directory,
                        waypoint_name="mission",
                        view_name="first",
                        frame_bgr=frame_bgr,
                        evidence_paths=evidence_paths,
                    )

                if len(capture_records) == minimum_capture_count // 2:
                    save_evidence_frame(
                        evidence_directory=evidence_directory,
                        waypoint_name="mission",
                        view_name="middle",
                        frame_bgr=frame_bgr,
                        evidence_paths=evidence_paths,
                    )

                brightness = float(grayscale.mean())
                contrast = float(grayscale.std())
                entropy = image_entropy(grayscale)
                sharpness = float(
                    cv2.Laplacian(
                        grayscale,
                        cv2.CV_64F,
                    ).var()
                )
                non_black_ratio = float(np.mean(grayscale > 5))
                saturated_ratio = float(np.mean(grayscale >= 250))

                if last_capture_grayscale is None:
                    frame_difference = 0.0
                else:
                    frame_difference = float(
                        cv2.absdiff(
                            grayscale,
                            last_capture_grayscale,
                        ).mean()
                    )

                last_capture_grayscale = grayscale.copy()

                checksum = hashlib.sha256(image_bytes).hexdigest()

                unique_checksums.add(checksum)

                if state in {
                    STATE_NAVIGATE,
                    STATE_SCAN,
                }:
                    target = waypoints[target_waypoint_index]
                    target_name = target.name
                    target_x = target.x
                    target_z = target.z
                    distance_to_target = math.hypot(
                        target_x - float(current_position[0]),
                        target_z - float(current_position[1]),
                    )
                else:
                    target_name = "NONE"
                    target_x = float("nan")
                    target_z = float("nan")
                    distance_to_target = float("nan")

                record = {
                    "schema_version": 1,
                    "stage": "5A3",
                    "capture_index": len(capture_records),
                    "simulation_time_seconds": (simulation_time),
                    "mission_state": state,
                    "target_waypoint_index": (target_waypoint_index),
                    "target_waypoint_name": (target_name),
                    "target_x": target_x,
                    "target_z": target_z,
                    "distance_to_target_meters": (distance_to_target),
                    "active_scan_view": (active_scan_view),
                    "position_x": float(current_position[0]),
                    "position_z": float(current_position[1]),
                    "gps": gps_values,
                    "roll_pitch_yaw": rpy_values,
                    "orientation_quaternion": orientation_quaternion,
                    "roll_degrees": math.degrees(current_roll),
                    "pitch_degrees": math.degrees(current_pitch),
                    "compass": compass_values,
                    "world_heading_degrees": (math.degrees(current_world_heading)),
                    "left_motor_command": (left_command),
                    "right_motor_command": (right_command),
                    "left_wheel_position_radians": (left_wheel_position),
                    "right_wheel_position_radians": (right_wheel_position),
                    "maximum_proximity_ratio": (proximity_ratio),
                    "distance_sensor_count": len(distance_sensors),
                    "width": width,
                    "height": height,
                    "pixel_format": "BGRA",
                    "expected_byte_length": (expected_byte_length),
                    "actual_byte_length": len(image_bytes),
                    "sha256": checksum,
                    "brightness_mean": brightness,
                    "contrast_std": contrast,
                    "entropy_bits": entropy,
                    "sharpness_laplacian_variance": (sharpness),
                    "non_black_ratio": (non_black_ratio),
                    "saturated_ratio": (saturated_ratio),
                    "frame_difference_mean": (frame_difference),
                    "closed_loop_position_control": (True),
                    "closed_loop_heading_control": (True),
                    "cv_model_connected": False,
                    "policy_controls_motors": False,
                }

                telemetry_file.write(
                    json.dumps(
                        record,
                        sort_keys=True,
                    )
                    + "\n"
                )
                telemetry_file.flush()

                capture_records.append(record)

                if len(capture_records) % 25 == 0:
                    print(
                        "STAGE5A3_NAVIGATION_DIAGNOSTIC "
                        f"frames={len(capture_records)} "
                        f"state={state} "
                        f"target={target_name} "
                        f"distance={distance_to_target:.3f} "
                        f"x={float(current_position[0]):.3f} "
                        f"z={float(current_position[1]):.3f} "
                        f"heading="
                        f"{math.degrees(current_world_heading):.2f} "
                        f"roll={math.degrees(current_roll):.2f} "
                        f"pitch={math.degrees(current_pitch):.2f} "
                        f"left={left_command:.3f} "
                        f"right={right_command:.3f}",
                        flush=True,
                    )

                next_capture_time = simulation_time + capture_interval_seconds

                if len(capture_records) % 50 == 0:
                    print(
                        "STAGE5A3_CAPTURE_PROGRESS "
                        f"frames={len(capture_records)} "
                        f"unique={len(unique_checksums)} "
                        f"state={state} "
                        f"target={target_name}",
                        flush=True,
                    )

            if state == STATE_COMPLETE and len(capture_records) >= minimum_capture_count:
                left_motor.setVelocity(0.0)
                right_motor.setVelocity(0.0)

                _, final_frame_bgr, _ = read_camera_frame(
                    camera,
                    width=width,
                    height=height,
                )
                save_evidence_frame(
                    evidence_directory=evidence_directory,
                    waypoint_name="mission",
                    view_name="final",
                    frame_bgr=final_frame_bgr,
                    evidence_paths=evidence_paths,
                )

                if initial_position is None:
                    raise RuntimeError("Initial position was not recorded.")

                final_position = current_position.copy()

                returned_to_start_distance = float(
                    np.linalg.norm(final_position - initial_position)
                )

                timestamps = [
                    float(record["simulation_time_seconds"]) for record in capture_records
                ]

                timestamps_increasing = all(
                    current > previous
                    for previous, current in zip(
                        timestamps,
                        timestamps[1:],
                        strict=False,
                    )
                )

                frame_differences = [
                    float(record["frame_difference_mean"]) for record in capture_records[1:]
                ]

                summary = {
                    "schema_version": 1,
                    "stage": "5A3",
                    "runtime_completed": True,
                    "route_completed": (route_completed),
                    "route_name": str(route_config["route_name"]),
                    "waypoints_total": len(waypoints),
                    "waypoints_visited": len(visited_waypoint_names),
                    "visited_waypoint_names": (visited_waypoint_names),
                    "arrival_records": (arrival_records),
                    "returned_to_start": (returned_to_start_distance <= 0.60),
                    "returned_to_start_distance_meters": (returned_to_start_distance),
                    "route_completed_at_seconds": (route_completed_at),
                    "mission_duration_seconds": (simulation_time),
                    "robot_path_length_meters": (path_length),
                    "total_heading_change_degrees": (math.degrees(total_heading_change)),
                    "recovery_count": (recovery_count),
                    "obstacle_interventions": (obstacle_interventions),
                    "distance_sensor_count": len(distance_sensors),
                    "navigation_calibrated": True,
                    "calibration_displacement_meters": (calibration_displacement),
                    "heading_offset_degrees": (math.degrees(heading_offset)),
                    "turn_command_sign": (turn_command_sign),
                    "yaw_to_world_sign": (YAW_TO_WORLD_SIGN),
                    "forward_motor_sign": (FORWARD_MOTOR_SIGN),
                    "closed_loop_position_control": (True),
                    "closed_loop_heading_control": (True),
                    "motor_motion_source": ("closed_loop_waypoint_controller"),
                    "rear_support_model": ("dual_low_friction_passive_ball_casters"),
                    "camera_enabled": True,
                    "camera_name": CAMERA_NAME,
                    "width": width,
                    "height": height,
                    "pixel_format": "BGRA",
                    "expected_byte_length": (expected_byte_length),
                    "capture_count": len(capture_records),
                    "capture_interval_seconds": (capture_interval_seconds),
                    "unique_frame_checksums": len(unique_checksums),
                    "timestamps_strictly_increasing": (timestamps_increasing),
                    "mean_brightness": float(
                        np.mean([record["brightness_mean"] for record in capture_records])
                    ),
                    "mean_contrast": float(
                        np.mean([record["contrast_std"] for record in capture_records])
                    ),
                    "mean_entropy_bits": float(
                        np.mean([record["entropy_bits"] for record in capture_records])
                    ),
                    "mean_sharpness": float(
                        np.mean(
                            [record["sharpness_laplacian_variance"] for record in capture_records]
                        )
                    ),
                    "minimum_non_black_ratio": float(
                        np.min([record["non_black_ratio"] for record in capture_records])
                    ),
                    "maximum_saturated_ratio": float(
                        np.max([record["saturated_ratio"] for record in capture_records])
                    ),
                    "mean_frame_difference": float(np.mean(frame_differences)),
                    "maximum_frame_difference": float(np.max(frame_differences)),
                    "evidence_frames": (evidence_paths),
                    "timeline_evidence": {
                        "first": str(evidence_directory / "mission_first.png"),
                        "middle": str(evidence_directory / "mission_middle.png"),
                        "final": str(evidence_directory / "mission_final.png"),
                    },
                    "telemetry_path": str(telemetry_path),
                    "collision_free_claimed": False,
                    "cv_model_connected": False,
                    "policy_controls_motors": False,
                }

                write_json(
                    summary_path,
                    summary,
                )

                completion_marker_path.write_text(
                    "STAGE5A3_COMPLETE\n",
                    encoding="utf-8",
                    newline="\n",
                )

                print(
                    "STAGE5A3_MISSION_COMPLETE "
                    f"waypoints={len(visited_waypoint_names)}/"
                    f"{len(waypoints)} "
                    f"frames={len(capture_records)} "
                    f"path_length={path_length:.3f} "
                    f"return_distance="
                    f"{returned_to_start_distance:.3f}",
                    flush=True,
                )

                return


def write_failure(
    exception: BaseException,
) -> None:
    try:
        root = output_directory()
        root.mkdir(
            parents=True,
            exist_ok=True,
        )

        payload = {
            "stage": "5A3",
            "runtime_completed": False,
            "exception_type": (type(exception).__name__),
            "exception_message": str(exception),
            "traceback": traceback.format_exc(),
        }

        write_json(
            root / "stage5a3_failure.json",
            payload,
        )

        (root / "stage5a3_failure.marker").write_text(
            "STAGE5A3_FAILURE\n",
            encoding="utf-8",
            newline="\n",
        )
    except Exception:
        print(
            traceback.format_exc(),
            flush=True,
        )


if __name__ == "__main__":
    try:
        main()
    except BaseException as exception:
        write_failure(exception)

        print(
            f"STAGE5A3_CONTROLLER_FAILURE {type(exception).__name__}: {exception}",
            flush=True,
        )

        raise
