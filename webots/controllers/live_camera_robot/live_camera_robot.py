from __future__ import annotations

import hashlib
import json
import math
import os
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from controller import Robot

CAMERA_NAME = "inspection camera"
GPS_NAME = "gps"
INERTIAL_UNIT_NAME = "inertial unit"
COMPASS_NAME = "compass"
LEFT_MOTOR_NAME = "left wheel motor"
RIGHT_MOTOR_NAME = "right wheel motor"

CAPTURE_COUNT = 120
CAPTURE_STRIDE_STEPS = 4

EVIDENCE_FRAME_INDICES = {
    0: "frame_000_start.png",
    59: "frame_059_middle.png",
    119: "frame_119_end.png",
}


@dataclass(frozen=True)
class MotionPhase:
    name: str
    duration: float
    left_velocity: float
    right_velocity: float


PHASES = (
    MotionPhase(
        "SYSTEM_CHECK",
        1.5,
        0.0,
        0.0,
    ),
    MotionPhase(
        "SAFE_WALKWAY_EAST",
        8.5,
        -1.2,
        -1.2,
    ),
    MotionPhase(
        "TURN_TO_CHECKPOINT",
        1.4,
        -0.3,
        0.3,
    ),
    MotionPhase(
        "APPROACH_INSPECTION_ZONE",
        6.0,
        -1.0,
        -1.0,
    ),
    MotionPhase(
        "SCAN_LEFT",
        1.4,
        -0.3,
        0.3,
    ),
    MotionPhase(
        "RETURN_ROUTE_WEST",
        8.5,
        -1.2,
        -1.2,
    ),
    MotionPhase(
        "TURN_TO_HOME_LANE",
        1.4,
        -0.3,
        0.3,
    ),
    MotionPhase(
        "RETURN_TO_START",
        6.0,
        -1.0,
        -1.0,
    ),
    MotionPhase(
        "FINAL_ALIGNMENT",
        1.4,
        -0.3,
        0.3,
    ),
    MotionPhase(
        "SAFE_STOP",
        2.0,
        0.0,
        0.0,
    ),
)

CYCLE_DURATION = sum(phase.duration for phase in PHASES)
MOTOR_COMMAND_SLEW_RATE_RADIANS_PER_SECOND = 0.5


def move_toward(current: float, target: float, maximum_delta: float) -> float:
    if target > current:
        return min(target, current + maximum_delta)
    return max(target, current - maximum_delta)


def project_root() -> Path:
    value = os.environ.get("RISK_AWARE_PROJECT_ROOT")

    if not value:
        raise RuntimeError("RISK_AWARE_PROJECT_ROOT is not set.")

    return Path(value).resolve()


def output_directory() -> Path:
    return project_root() / "webots" / "logs" / "stage5a_live_camera"


def resolve_phase(
    simulation_time: float,
) -> MotionPhase:
    cycle_time = simulation_time % CYCLE_DURATION

    elapsed = 0.0

    for phase in PHASES:
        elapsed += phase.duration

        if cycle_time < elapsed:
            return phase

    return PHASES[-1]


def finite_list(
    values: list[float],
    *,
    name: str,
) -> list[float]:
    converted = [float(value) for value in values]

    if not all(math.isfinite(value) for value in converted):
        raise RuntimeError(f"{name} contains a non-finite value.")

    return converted


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


def compute_path_length(
    positions: list[list[float]],
) -> float:
    if len(positions) < 2:
        return 0.0

    array = np.asarray(
        positions,
        dtype=np.float64,
    )

    differences = np.diff(
        array,
        axis=0,
    )

    return float(
        np.linalg.norm(
            differences,
            axis=1,
        ).sum()
    )


def compute_displacement(
    positions: list[list[float]],
) -> float:
    if len(positions) < 2:
        return 0.0

    first = np.asarray(
        positions[0],
        dtype=np.float64,
    )

    last = np.asarray(
        positions[-1],
        dtype=np.float64,
    )

    return float(np.linalg.norm(last - first))


def main() -> None:
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

    telemetry_path = root / "stage5a_camera_frames.jsonl"

    summary_path = root / "stage5a_camera_summary.json"

    completion_marker_path = root / "stage5a_complete.marker"

    robot = Robot()

    time_step = int(robot.getBasicTimeStep())

    camera = robot.getDevice(CAMERA_NAME)

    gps = robot.getDevice(GPS_NAME)

    inertial_unit = robot.getDevice(INERTIAL_UNIT_NAME)

    compass = robot.getDevice(COMPASS_NAME)

    left_motor = robot.getDevice(LEFT_MOTOR_NAME)

    right_motor = robot.getDevice(RIGHT_MOTOR_NAME)

    devices = {
        CAMERA_NAME: camera,
        GPS_NAME: gps,
        INERTIAL_UNIT_NAME: inertial_unit,
        COMPASS_NAME: compass,
        LEFT_MOTOR_NAME: left_motor,
        RIGHT_MOTOR_NAME: right_motor,
    }

    for name, device in devices.items():
        if device is None:
            raise RuntimeError(f"Webots device was not found: {name}")

    camera.enable(time_step)
    gps.enable(time_step)
    inertial_unit.enable(time_step)
    compass.enable(time_step)

    left_motor.setPosition(float("inf"))

    right_motor.setPosition(float("inf"))

    left_motor.setVelocity(0.0)
    right_motor.setVelocity(0.0)

    width = int(camera.getWidth())

    height = int(camera.getHeight())

    expected_byte_length = width * height * 4

    if (width, height) != (640, 360):
        raise RuntimeError(f"Unexpected camera resolution: {width}x{height}")

    records: list[dict[str, Any]] = []
    checksums: set[str] = set()
    positions: list[list[float]] = []
    evidence_paths: list[str] = []

    previous_grayscale: np.ndarray | None = None

    step_index = 0
    capture_index = 0
    completed = False
    applied_left_velocity = 0.0
    applied_right_velocity = 0.0

    print(
        "STAGE5A_CAMERA_CONTROLLER_READY "
        f"resolution={width}x{height} "
        f"expected_bytes={expected_byte_length}",
        flush=True,
    )

    with telemetry_path.open(
        "w",
        encoding="utf-8",
        newline="\n",
    ) as telemetry_file:
        while robot.step(time_step) != -1:
            simulation_time = float(robot.getTime())

            if completed:
                left_motor.setVelocity(0.0)
                right_motor.setVelocity(0.0)
                continue

            phase = resolve_phase(simulation_time)

            maximum_delta = MOTOR_COMMAND_SLEW_RATE_RADIANS_PER_SECOND * time_step / 1000.0
            applied_left_velocity = move_toward(
                applied_left_velocity, phase.left_velocity, maximum_delta
            )
            applied_right_velocity = move_toward(
                applied_right_velocity, phase.right_velocity, maximum_delta
            )
            left_motor.setVelocity(applied_left_velocity)
            right_motor.setVelocity(applied_right_velocity)

            if step_index % CAPTURE_STRIDE_STEPS != 0:
                step_index += 1
                continue

            raw_image = camera.getImage()

            if raw_image is None:
                raise RuntimeError("Camera returned no image.")

            image_bytes = bytes(raw_image)

            actual_byte_length = len(image_bytes)

            if actual_byte_length != expected_byte_length:
                raise RuntimeError(
                    "Unexpected camera byte length. "
                    f"Expected {expected_byte_length}, "
                    f"received {actual_byte_length}."
                )

            frame_bgra = np.frombuffer(
                image_bytes,
                dtype=np.uint8,
            ).reshape(
                height,
                width,
                4,
            )

            frame_bgr = frame_bgra[
                :,
                :,
                :3,
            ].copy()

            grayscale = cv2.cvtColor(
                frame_bgr,
                cv2.COLOR_BGR2GRAY,
            )

            brightness = float(grayscale.mean())

            contrast = float(grayscale.std())

            non_black_ratio = float(np.mean(grayscale > 5))

            saturated_ratio = float(np.mean(grayscale >= 250))

            entropy = image_entropy(grayscale)

            sharpness = float(
                cv2.Laplacian(
                    grayscale,
                    cv2.CV_64F,
                ).var()
            )

            if previous_grayscale is None:
                frame_difference = 0.0
            else:
                frame_difference = float(
                    cv2.absdiff(
                        grayscale,
                        previous_grayscale,
                    ).mean()
                )

            previous_grayscale = grayscale.copy()

            checksum = hashlib.sha256(image_bytes).hexdigest()

            checksums.add(checksum)

            gps_values = finite_list(
                list(gps.getValues()),
                name="gps",
            )

            rpy_values = finite_list(
                list(inertial_unit.getRollPitchYaw()),
                name="inertial_unit",
            )

            compass_values = finite_list(
                list(compass.getValues()),
                name="compass",
            )

            positions.append(gps_values)

            record = {
                "schema_version": 1,
                "stage": "5A1",
                "capture_index": capture_index,
                "simulation_step": step_index,
                "simulation_time_seconds": (simulation_time),
                "motion_phase": phase.name,
                "camera_name": CAMERA_NAME,
                "width": width,
                "height": height,
                "pixel_format": "BGRA",
                "expected_byte_length": (expected_byte_length),
                "actual_byte_length": (actual_byte_length),
                "sha256": checksum,
                "brightness_mean": brightness,
                "contrast_std": contrast,
                "entropy_bits": entropy,
                "sharpness_laplacian_variance": (sharpness),
                "non_black_ratio": (non_black_ratio),
                "saturated_ratio": (saturated_ratio),
                "frame_difference_mean": (frame_difference),
                "gps": gps_values,
                "roll_pitch_yaw": rpy_values,
                "compass": compass_values,
                "yaw_degrees": float(math.degrees(rpy_values[2])),
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

            records.append(record)

            evidence_filename = EVIDENCE_FRAME_INDICES.get(capture_index)

            if evidence_filename is not None:
                evidence_path = evidence_directory / evidence_filename

                saved = cv2.imwrite(
                    str(evidence_path),
                    frame_bgr,
                )

                if not saved:
                    raise RuntimeError(f"Could not save evidence frame: {evidence_path}")

                evidence_paths.append(str(evidence_path))

            capture_index += 1
            step_index += 1

            if capture_index % 20 == 0:
                print(
                    "STAGE5A_CAMERA_PROGRESS "
                    f"captured={capture_index}/"
                    f"{CAPTURE_COUNT} "
                    f"unique={len(checksums)} "
                    f"phase={phase.name}",
                    flush=True,
                )

            if capture_index < CAPTURE_COUNT:
                continue

            left_motor.setVelocity(0.0)
            right_motor.setVelocity(0.0)

            timestamps = [float(record["simulation_time_seconds"]) for record in records]

            timestamps_increasing = all(
                current > previous
                for previous, current in zip(
                    timestamps,
                    timestamps[1:],
                    strict=False,
                )
            )

            frame_differences = [float(record["frame_difference_mean"]) for record in records[1:]]

            summary = {
                "schema_version": 1,
                "stage": "5A1",
                "runtime_completed": True,
                "camera_enabled": True,
                "camera_name": CAMERA_NAME,
                "width": width,
                "height": height,
                "pixel_format": "BGRA",
                "expected_byte_length": (expected_byte_length),
                "capture_count": len(records),
                "capture_stride_steps": (CAPTURE_STRIDE_STEPS),
                "unique_frame_checksums": (len(checksums)),
                "timestamps_strictly_increasing": (timestamps_increasing),
                "mean_brightness": float(
                    np.mean([record["brightness_mean"] for record in records])
                ),
                "mean_contrast": float(np.mean([record["contrast_std"] for record in records])),
                "mean_entropy_bits": float(np.mean([record["entropy_bits"] for record in records])),
                "mean_sharpness": float(
                    np.mean([record["sharpness_laplacian_variance"] for record in records])
                ),
                "minimum_non_black_ratio": float(
                    np.min([record["non_black_ratio"] for record in records])
                ),
                "maximum_saturated_ratio": float(
                    np.max([record["saturated_ratio"] for record in records])
                ),
                "mean_frame_difference": float(np.mean(frame_differences)),
                "maximum_frame_difference": float(np.max(frame_differences)),
                "robot_path_length_meters": (compute_path_length(positions)),
                "robot_displacement_meters": (compute_displacement(positions)),
                "evidence_frames": evidence_paths,
                "telemetry_path": str(telemetry_path),
                "cv_model_connected": False,
                "policy_controls_motors": False,
                "motor_motion_source": ("scripted_stage5a_validation"),
            }

            write_json(
                summary_path,
                summary,
            )

            completion_marker_path.write_text(
                "STAGE5A_COMPLETE\n",
                encoding="utf-8",
                newline="\n",
            )

            print(
                "STAGE5A_CAMERA_COMPLETE "
                f"captured={len(records)} "
                f"unique={len(checksums)} "
                f"path_length="
                f"{summary['robot_path_length_meters']:.3f}",
                flush=True,
            )

            completed = True


def write_failure(
    exception: BaseException,
) -> None:
    try:
        root = output_directory()

        root.mkdir(
            parents=True,
            exist_ok=True,
        )

        failure_payload = {
            "stage": "5A1",
            "runtime_completed": False,
            "exception_type": (type(exception).__name__),
            "exception_message": str(exception),
            "traceback": traceback.format_exc(),
        }

        write_json(
            root / "stage5a_failure.json",
            failure_payload,
        )

        (root / "stage5a_failure.marker").write_text(
            "STAGE5A_FAILURE\n",
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
            f"STAGE5A_CAMERA_FAILURE {type(exception).__name__}: {exception}",
            flush=True,
        )

        raise
