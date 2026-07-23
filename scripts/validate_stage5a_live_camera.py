from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Any

EXPECTED_CAPTURE_COUNT = 120
EXPECTED_WIDTH = 640
EXPECTED_HEIGHT = 360
EXPECTED_BYTE_LENGTH = EXPECTED_WIDTH * EXPECTED_HEIGHT * 4

MINIMUM_UNIQUE_CHECKSUMS = 20
MINIMUM_MEAN_BRIGHTNESS = 10.0
MAXIMUM_MEAN_BRIGHTNESS = 245.0
MINIMUM_MEAN_CONTRAST = 8.0
MINIMUM_MEAN_ENTROPY_BITS = 2.0
MINIMUM_MEAN_SHARPNESS = 5.0
MINIMUM_NON_BLACK_RATIO = 0.25
MAXIMUM_SATURATED_RATIO = 0.80
MINIMUM_MEAN_FRAME_DIFFERENCE = 0.25
MINIMUM_ROBOT_PATH_LENGTH_METERS = 0.50


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

    output_directory = project / "webots" / "logs" / "stage5a_live_camera"

    telemetry_path = output_directory / "stage5a_camera_frames.jsonl"

    summary_path = output_directory / "stage5a_camera_summary.json"

    completion_marker_path = output_directory / "stage5a_complete.marker"

    failure_path = output_directory / "stage5a_failure.json"

    timeout_marker_path = output_directory / "stage5a_timeout.marker"

    validation_path = output_directory / "stage5a_validation_report.json"

    require(
        not failure_path.exists(),
        (f"The Stage 5A runtime created a failure report: {failure_path}"),
    )

    require(
        not timeout_marker_path.exists(),
        "The Stage 5A runtime timed out.",
    )

    require(
        completion_marker_path.is_file(),
        "The Stage 5A completion marker is missing.",
    )

    require(
        telemetry_path.is_file(),
        "The Stage 5A telemetry file is missing.",
    )

    records = [
        json.loads(line)
        for line in telemetry_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    summary = load_json(summary_path)

    require(
        len(records) == EXPECTED_CAPTURE_COUNT,
        (
            "Unexpected telemetry record count. "
            f"Expected {EXPECTED_CAPTURE_COUNT}, "
            f"received {len(records)}."
        ),
    )

    capture_indices = [int(record["capture_index"]) for record in records]

    require(
        capture_indices == list(range(EXPECTED_CAPTURE_COUNT)),
        "Camera capture indices are not contiguous.",
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
        "Camera timestamps are not increasing.",
    )

    required_record_keys = {
        "capture_index",
        "simulation_step",
        "simulation_time_seconds",
        "motion_phase",
        "camera_name",
        "width",
        "height",
        "pixel_format",
        "expected_byte_length",
        "actual_byte_length",
        "sha256",
        "brightness_mean",
        "contrast_std",
        "entropy_bits",
        "sharpness_laplacian_variance",
        "non_black_ratio",
        "saturated_ratio",
        "frame_difference_mean",
        "gps",
        "roll_pitch_yaw",
        "compass",
        "yaw_degrees",
        "cv_model_connected",
        "policy_controls_motors",
    }

    for record in records:
        require(
            required_record_keys.issubset(record),
            ("A telemetry record is missing required fields."),
        )

        require(
            int(record["width"]) == EXPECTED_WIDTH,
            "A frame has an invalid width.",
        )

        require(
            int(record["height"]) == EXPECTED_HEIGHT,
            "A frame has an invalid height.",
        )

        require(
            record["pixel_format"] == "BGRA",
            "A frame has an invalid pixel format.",
        )

        require(
            int(record["expected_byte_length"]) == EXPECTED_BYTE_LENGTH,
            ("A frame has an invalid expected byte length."),
        )

        require(
            int(record["actual_byte_length"]) == EXPECTED_BYTE_LENGTH,
            ("A frame has an invalid actual byte length."),
        )

        require(
            len(str(record["sha256"])) == 64,
            "A frame has an invalid SHA-256 hash.",
        )

        require(
            len(record["gps"]) == 3,
            "A frame has invalid GPS telemetry.",
        )

        require(
            len(record["roll_pitch_yaw"]) == 3,
            "A frame has invalid IMU telemetry.",
        )

        require(
            len(record["compass"]) == 3,
            "A frame has invalid compass telemetry.",
        )

        require(
            all(
                math.isfinite(float(value))
                for value in (record["gps"] + record["roll_pitch_yaw"] + record["compass"])
            ),
            ("A frame contains non-finite sensor telemetry."),
        )

        require(
            not bool(record["cv_model_connected"]),
            ("A CV model must not be connected during Stage 5A."),
        )

        require(
            not bool(record["policy_controls_motors"]),
            ("The policy must not control motors during Stage 5A."),
        )

    checksums = {str(record["sha256"]) for record in records}

    require(
        len(checksums) >= MINIMUM_UNIQUE_CHECKSUMS,
        (f"Insufficient temporal image variation. Unique frames: {len(checksums)}."),
    )

    mean_brightness = float(summary["mean_brightness"])

    mean_contrast = float(summary["mean_contrast"])

    mean_entropy = float(summary["mean_entropy_bits"])

    mean_sharpness = float(summary["mean_sharpness"])

    minimum_non_black_ratio = float(summary["minimum_non_black_ratio"])

    maximum_saturated_ratio = float(summary["maximum_saturated_ratio"])

    mean_frame_difference = float(summary["mean_frame_difference"])

    path_length = float(summary["robot_path_length_meters"])

    require(
        (MINIMUM_MEAN_BRIGHTNESS <= mean_brightness <= MAXIMUM_MEAN_BRIGHTNESS),
        (f"Mean brightness is outside the valid range: {mean_brightness:.4f}."),
    )

    require(
        mean_contrast >= MINIMUM_MEAN_CONTRAST,
        (f"Camera contrast is too low: {mean_contrast:.4f}."),
    )

    require(
        mean_entropy >= MINIMUM_MEAN_ENTROPY_BITS,
        (f"Camera entropy is too low: {mean_entropy:.4f}."),
    )

    require(
        mean_sharpness >= MINIMUM_MEAN_SHARPNESS,
        (f"Camera sharpness is too low: {mean_sharpness:.4f}."),
    )

    require(
        minimum_non_black_ratio >= MINIMUM_NON_BLACK_RATIO,
        (f"Camera frames contain too many black pixels: {minimum_non_black_ratio:.4f}."),
    )

    require(
        maximum_saturated_ratio <= MAXIMUM_SATURATED_RATIO,
        (f"Camera frames contain too many saturated pixels: {maximum_saturated_ratio:.4f}."),
    )

    require(
        mean_frame_difference >= MINIMUM_MEAN_FRAME_DIFFERENCE,
        (f"Temporal frame difference is too low: {mean_frame_difference:.4f}."),
    )

    require(
        path_length >= MINIMUM_ROBOT_PATH_LENGTH_METERS,
        (f"Robot path length is too short: {path_length:.4f} meters."),
    )

    evidence_paths = [Path(path) for path in summary["evidence_frames"]]

    require(
        len(evidence_paths) == 3,
        ("Expected exactly three evidence frames."),
    )

    for evidence_path in evidence_paths:
        require(
            evidence_path.is_file(),
            (f"Evidence frame is missing: {evidence_path}"),
        )

        require(
            evidence_path.stat().st_size > 1000,
            (f"Evidence frame is unexpectedly small: {evidence_path}"),
        )

    validation_report = {
        "schema_version": 1,
        "stage": "5A2",
        "runtime_verified": True,
        "camera_enabled": True,
        "camera_name": (summary["camera_name"]),
        "resolution": (f"{EXPECTED_WIDTH}x{EXPECTED_HEIGHT}"),
        "pixel_format": "BGRA",
        "capture_count": len(records),
        "expected_byte_length": (EXPECTED_BYTE_LENGTH),
        "unique_frame_checksums": (len(checksums)),
        "timestamps_strictly_increasing": True,
        "mean_brightness": mean_brightness,
        "mean_contrast": mean_contrast,
        "mean_entropy_bits": mean_entropy,
        "mean_sharpness": mean_sharpness,
        "minimum_non_black_ratio": (minimum_non_black_ratio),
        "maximum_saturated_ratio": (maximum_saturated_ratio),
        "mean_frame_difference": (mean_frame_difference),
        "robot_path_length_meters": (path_length),
        "evidence_frames": [str(path) for path in evidence_paths],
        "gps_synchronized": True,
        "imu_synchronized": True,
        "compass_synchronized": True,
        "cv_model_connected": False,
        "policy_controls_motors": False,
        "validation_thresholds": {
            "minimum_unique_checksums": (MINIMUM_UNIQUE_CHECKSUMS),
            "minimum_mean_brightness": (MINIMUM_MEAN_BRIGHTNESS),
            "maximum_mean_brightness": (MAXIMUM_MEAN_BRIGHTNESS),
            "minimum_mean_contrast": (MINIMUM_MEAN_CONTRAST),
            "minimum_mean_entropy_bits": (MINIMUM_MEAN_ENTROPY_BITS),
            "minimum_mean_sharpness": (MINIMUM_MEAN_SHARPNESS),
            "minimum_non_black_ratio": (MINIMUM_NON_BLACK_RATIO),
            "maximum_saturated_ratio": (MAXIMUM_SATURATED_RATIO),
            "minimum_mean_frame_difference": (MINIMUM_MEAN_FRAME_DIFFERENCE),
            "minimum_robot_path_length_meters": (MINIMUM_ROBOT_PATH_LENGTH_METERS),
        },
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
