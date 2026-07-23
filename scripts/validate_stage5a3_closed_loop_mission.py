from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Any

import cv2

MINIMUM_CAPTURE_COUNT = 180
MINIMUM_UNIQUE_CHECKSUMS = 100
MINIMUM_PATH_LENGTH_METERS = 8.0
MAXIMUM_RETURN_DISTANCE_METERS = 0.60
MINIMUM_HEADING_CHANGE_DEGREES = 180.0
MINIMUM_EVIDENCE_FRAME_COUNT = 10
MAXIMUM_ARRIVAL_DISTANCE_METERS = 0.45
MAXIMUM_RECOVERY_COUNT = 3
MINIMUM_CALIBRATION_DISPLACEMENT_METERS = 0.10
MINIMUM_WHEEL_POSITION_SPAN_RADIANS = 10.0
MAXIMUM_TELEMETRY_TILT_DEGREES = 10.0
MINIMUM_MEAN_BRIGHTNESS = 10.0
MAXIMUM_MEAN_BRIGHTNESS = 245.0
MINIMUM_MEAN_CONTRAST = 8.0
MINIMUM_MEAN_ENTROPY_BITS = 2.0
MINIMUM_MEAN_SHARPNESS = 5.0
MINIMUM_NON_BLACK_RATIO = 0.25
MINIMUM_MEAN_FRAME_DIFFERENCE = 0.20


def require(
    condition: bool,
    message: str,
) -> None:
    if not condition:
        raise RuntimeError(message)


def load_json(
    path: Path,
) -> dict[str, Any]:
    require(
        path.is_file(),
        f"Required JSON file is missing: {path}",
    )

    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    project = Path(sys.argv[1]).resolve()

    output_directory = project / "webots" / "logs" / "stage5a3_closed_loop"

    telemetry_path = output_directory / "stage5a3_mission_telemetry.jsonl"

    summary_path = output_directory / "stage5a3_mission_summary.json"

    completion_marker_path = output_directory / "stage5a3_complete.marker"

    failure_path = output_directory / "stage5a3_failure.json"

    timeout_marker_path = output_directory / "stage5a3_timeout.marker"

    validation_path = output_directory / "stage5a3_validation_report.json"

    require(
        not failure_path.exists(),
        (f"The Stage 5A3 runtime created a failure report: {failure_path}"),
    )

    require(
        not timeout_marker_path.exists(),
        "The Stage 5A3 runtime timed out.",
    )

    require(
        completion_marker_path.is_file(),
        "The Stage 5A3 completion marker is missing.",
    )

    require(
        telemetry_path.is_file(),
        "The Stage 5A3 telemetry file is missing.",
    )

    records = [
        json.loads(line)
        for line in telemetry_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    summary = load_json(summary_path)

    require(
        len(records) >= MINIMUM_CAPTURE_COUNT,
        (f"Insufficient closed-loop camera frames. Received {len(records)}."),
    )

    capture_indices = [int(record["capture_index"]) for record in records]

    require(
        capture_indices == list(range(len(records))),
        "Capture indices are not contiguous.",
    )

    timestamps = [float(record["simulation_time_seconds"]) for record in records]

    require(
        all(
            current > previous
            for previous, current in zip(
                timestamps,
                timestamps[1:],
                strict=False,
            )
        ),
        "Telemetry timestamps are not increasing.",
    )

    required_record_keys = {
        "mission_state",
        "target_waypoint_name",
        "position_x",
        "position_z",
        "gps",
        "roll_pitch_yaw",
        "compass",
        "world_heading_degrees",
        "left_motor_command",
        "right_motor_command",
        "left_wheel_position_radians",
        "right_wheel_position_radians",
        "roll_degrees",
        "pitch_degrees",
        "maximum_proximity_ratio",
        "distance_sensor_count",
        "width",
        "height",
        "pixel_format",
        "actual_byte_length",
        "sha256",
        "brightness_mean",
        "contrast_std",
        "entropy_bits",
        "sharpness_laplacian_variance",
        "frame_difference_mean",
        "closed_loop_position_control",
        "closed_loop_heading_control",
        "cv_model_connected",
        "policy_controls_motors",
    }

    for record in records:
        require(
            required_record_keys.issubset(record),
            "A telemetry record is incomplete.",
        )

        require(
            int(record["width"]) == 640,
            "A telemetry frame has an invalid width.",
        )

        require(
            int(record["height"]) == 360,
            "A telemetry frame has an invalid height.",
        )

        require(
            record["pixel_format"] == "BGRA",
            "A telemetry frame has an invalid format.",
        )

        require(
            int(record["actual_byte_length"]) == 640 * 360 * 4,
            "A telemetry frame has an invalid byte length.",
        )

        require(
            len(str(record["sha256"])) == 64,
            "A telemetry frame has an invalid checksum.",
        )

        require(
            len(record["gps"]) == 3,
            "A telemetry record has invalid GPS data.",
        )

        require(
            len(record["roll_pitch_yaw"]) == 3,
            "A telemetry record has invalid IMU data.",
        )

        require(
            len(record["compass"]) == 3,
            "A telemetry record has invalid compass data.",
        )

        require(
            all(
                math.isfinite(float(value))
                for value in (record["gps"] + record["roll_pitch_yaw"] + record["compass"])
            ),
            "A telemetry record contains non-finite sensor data.",
        )

        require(
            all(
                math.isfinite(float(value))
                for value in (
                    record["left_wheel_position_radians"],
                    record["right_wheel_position_radians"],
                    record["roll_degrees"],
                    record["pitch_degrees"],
                )
            ),
            "A telemetry record contains non-finite wheel or tilt data.",
        )

        require(
            abs(float(record["roll_degrees"])) <= MAXIMUM_TELEMETRY_TILT_DEGREES,
            "A telemetry record exceeds the roll limit.",
        )

        require(
            abs(float(record["pitch_degrees"])) <= MAXIMUM_TELEMETRY_TILT_DEGREES,
            "A telemetry record exceeds the pitch limit.",
        )

        require(
            bool(record["closed_loop_position_control"]),
            "Position control is not marked closed-loop.",
        )

        require(
            bool(record["closed_loop_heading_control"]),
            "Heading control is not marked closed-loop.",
        )

        require(
            not bool(record["cv_model_connected"]),
            "A CV model must remain disabled in Stage 5A3.",
        )

        require(
            not bool(record["policy_controls_motors"]),
            "Policy motor control must remain disabled in Stage 5A3.",
        )

    left_wheel_positions = [float(record["left_wheel_position_radians"]) for record in records]

    right_wheel_positions = [float(record["right_wheel_position_radians"]) for record in records]

    maximum_roll_degrees = max(abs(float(record["roll_degrees"])) for record in records)
    maximum_pitch_degrees = max(abs(float(record["pitch_degrees"])) for record in records)

    left_wheel_span = max(left_wheel_positions) - min(left_wheel_positions)

    right_wheel_span = max(right_wheel_positions) - min(right_wheel_positions)

    require(
        left_wheel_span >= MINIMUM_WHEEL_POSITION_SPAN_RADIANS,
        (f"Left wheel encoder span is too small: {left_wheel_span:.4f} radians."),
    )

    require(
        right_wheel_span >= MINIMUM_WHEEL_POSITION_SPAN_RADIANS,
        (f"Right wheel encoder span is too small: {right_wheel_span:.4f} radians."),
    )

    checksums = {str(record["sha256"]) for record in records}

    require(
        len(checksums) >= MINIMUM_UNIQUE_CHECKSUMS,
        (f"Insufficient temporal frame variation. Unique frames: {len(checksums)}."),
    )

    require(
        bool(summary["runtime_completed"]),
        "runtime_completed is not true.",
    )

    require(
        bool(summary["route_completed"]),
        "The closed-loop route did not complete.",
    )

    require(
        int(summary["waypoints_visited"]) == int(summary["waypoints_total"]),
        "Not all configured waypoints were visited.",
    )

    visited_names = list(summary["visited_waypoint_names"])

    require(
        len(visited_names) == int(summary["waypoints_total"]),
        "The waypoint visit log is incomplete.",
    )

    require(
        bool(summary["returned_to_start"]),
        "The robot did not return to the start area.",
    )

    return_distance = float(summary["returned_to_start_distance_meters"])

    require(
        return_distance <= MAXIMUM_RETURN_DISTANCE_METERS,
        (f"The return-to-start distance is too large: {return_distance:.4f} meters."),
    )

    path_length = float(summary["robot_path_length_meters"])

    require(
        path_length >= MINIMUM_PATH_LENGTH_METERS,
        (f"The closed-loop route is too short: {path_length:.4f} meters."),
    )

    heading_change = float(summary["total_heading_change_degrees"])

    require(
        heading_change >= MINIMUM_HEADING_CHANGE_DEGREES,
        (f"The route did not include sufficient heading change: {heading_change:.4f} degrees."),
    )

    require(
        bool(summary["navigation_calibrated"]),
        "Navigation calibration did not complete.",
    )

    calibration_displacement = float(summary["calibration_displacement_meters"])

    require(
        calibration_displacement >= MINIMUM_CALIBRATION_DISPLACEMENT_METERS,
        (f"Calibration displacement is too small: {calibration_displacement:.4f} meters."),
    )

    recovery_count = int(summary["recovery_count"])

    require(
        recovery_count <= MAXIMUM_RECOVERY_COUNT,
        "The mission required too many recoveries.",
    )

    require(
        summary["motor_motion_source"] == "closed_loop_waypoint_controller",
        "The motor motion source is incorrect.",
    )

    require(
        summary["rear_support_model"] == "dual_low_friction_passive_ball_casters",
        "The rear support model is incorrect.",
    )

    require(
        bool(summary["closed_loop_position_control"]),
        "Closed-loop position control was not verified.",
    )

    require(
        bool(summary["closed_loop_heading_control"]),
        "Closed-loop heading control was not verified.",
    )

    arrival_records = list(summary["arrival_records"])

    require(
        len(arrival_records) == int(summary["waypoints_total"]) - 1,
        "The arrival record count is invalid.",
    )

    maximum_arrival_distance = max(
        float(record["arrival_distance_meters"]) for record in arrival_records
    )

    require(
        maximum_arrival_distance <= MAXIMUM_ARRIVAL_DISTANCE_METERS,
        (f"A waypoint arrival exceeded tolerance: {maximum_arrival_distance:.4f} meters."),
    )

    evidence_paths = [Path(path) for path in summary["evidence_frames"]]

    require(
        len(evidence_paths) >= MINIMUM_EVIDENCE_FRAME_COUNT,
        (f"Insufficient zone evidence frames: {len(evidence_paths)}."),
    )

    for path in evidence_paths:
        require(
            path.is_file(),
            f"Evidence frame is missing: {path}",
        )

        require(
            path.stat().st_size > 1000,
            f"Evidence frame is unexpectedly small: {path}",
        )

        require(
            cv2.imread(str(path)) is not None,
            f"Evidence frame is not readable: {path}",
        )

    timeline_evidence = {name: Path(path) for name, path in summary["timeline_evidence"].items()}
    require(
        set(timeline_evidence) == {"first", "middle", "final"},
        "First, middle, and final timeline evidence is required.",
    )
    for path in timeline_evidence.values():
        require(path in evidence_paths, f"Timeline evidence was not registered: {path}")

    mean_brightness = float(summary["mean_brightness"])
    mean_contrast = float(summary["mean_contrast"])
    mean_entropy = float(summary["mean_entropy_bits"])
    mean_sharpness = float(summary["mean_sharpness"])
    minimum_non_black_ratio = float(summary["minimum_non_black_ratio"])
    mean_frame_difference = float(summary["mean_frame_difference"])

    require(
        MINIMUM_MEAN_BRIGHTNESS <= mean_brightness <= MAXIMUM_MEAN_BRIGHTNESS,
        "Mean brightness is outside the valid range.",
    )

    require(
        mean_contrast >= MINIMUM_MEAN_CONTRAST,
        "Mean contrast is too low.",
    )

    require(
        mean_entropy >= MINIMUM_MEAN_ENTROPY_BITS,
        "Mean entropy is too low.",
    )

    require(
        mean_sharpness >= MINIMUM_MEAN_SHARPNESS,
        "Mean sharpness is too low.",
    )

    require(
        minimum_non_black_ratio >= MINIMUM_NON_BLACK_RATIO,
        "The evidence contains too many black pixels.",
    )

    require(
        mean_frame_difference >= MINIMUM_MEAN_FRAME_DIFFERENCE,
        "Temporal frame variation is too low.",
    )

    require(
        not bool(summary["collision_free_claimed"]),
        ("Stage 5A3 must not claim collision-free operation without a dedicated contact sensor."),
    )

    require(
        not bool(summary["cv_model_connected"]),
        "A CV model must remain disabled.",
    )

    require(
        not bool(summary["policy_controls_motors"]),
        "Policy motor control must remain disabled.",
    )

    validation_report = {
        "schema_version": 1,
        "stage": "5A3",
        "runtime_verified": True,
        "route_completed": True,
        "closed_loop_position_control": True,
        "closed_loop_heading_control": True,
        "navigation_calibrated": True,
        "route_name": summary["route_name"],
        "waypoints_total": int(summary["waypoints_total"]),
        "waypoints_visited": int(summary["waypoints_visited"]),
        "visited_waypoint_names": visited_names,
        "maximum_arrival_distance_meters": (maximum_arrival_distance),
        "returned_to_start": True,
        "returned_to_start_distance_meters": (return_distance),
        "robot_path_length_meters": path_length,
        "total_heading_change_degrees": (heading_change),
        "recovery_count": recovery_count,
        "obstacle_interventions": int(summary["obstacle_interventions"]),
        "distance_sensor_count": int(summary["distance_sensor_count"]),
        "motor_motion_source": (summary["motor_motion_source"]),
        "rear_support_model": (summary["rear_support_model"]),
        "wheel_odometry_synchronized": True,
        "left_wheel_position_span_radians": (left_wheel_span),
        "right_wheel_position_span_radians": (right_wheel_span),
        "camera_enabled": True,
        "resolution": "640x360",
        "pixel_format": "BGRA",
        "capture_count": len(records),
        "unique_frame_checksums": len(checksums),
        "timestamps_strictly_increasing": True,
        "maximum_roll_degrees": maximum_roll_degrees,
        "maximum_pitch_degrees": maximum_pitch_degrees,
        "mean_brightness": mean_brightness,
        "mean_contrast": mean_contrast,
        "mean_entropy_bits": mean_entropy,
        "mean_sharpness": mean_sharpness,
        "minimum_non_black_ratio": (minimum_non_black_ratio),
        "mean_frame_difference": (mean_frame_difference),
        "evidence_frame_count": len(evidence_paths),
        "evidence_frames": [str(path) for path in evidence_paths],
        "timeline_evidence": {name: str(path) for name, path in timeline_evidence.items()},
        "gps_synchronized": True,
        "imu_synchronized": True,
        "compass_synchronized": True,
        "collision_free_claimed": False,
        "cv_model_connected": False,
        "policy_controls_motors": False,
    }

    validation_path.write_text(
        json.dumps(
            validation_report,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
        newline="\n",
    )

    print(
        json.dumps(
            validation_report,
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
