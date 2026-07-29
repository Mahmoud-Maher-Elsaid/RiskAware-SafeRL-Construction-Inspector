from __future__ import annotations

import argparse
import json
import math
import shutil
from pathlib import Path
from typing import Any

import cv2
import numpy as np

SOURCE_IMAGES = {
    "initial_first_person.png": "first_person_initial.jpg",
    "left_turn_first_person.png": "first_person_left_turn.jpg",
    "right_turn_first_person.png": "first_person_right_turn.jpg",
    "middle_first_person.png": "first_person_middle.jpg",
    "final_first_person.png": "first_person_final.jpg",
}


def image_metrics(image: np.ndarray) -> dict[str, Any]:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    height, width = gray.shape
    edges = cv2.Canny(gray, 60, 160)
    edge_density = float(np.count_nonzero(edges) / edges.size)
    brightness_standard_deviation = float(gray.std())
    quantized = (image // 16).reshape(-1, 3)
    _, counts = np.unique(quantized, axis=0, return_counts=True)
    dominant_color_ratio = float(counts.max() / counts.sum())

    sky_ratio = dominant_color_ratio
    lower_half = image[height // 2 :, :]
    lower_hsv = cv2.cvtColor(lower_half, cv2.COLOR_BGR2HSV)
    lower_sky = (
        (lower_hsv[:, :, 0] >= 90)
        & (lower_hsv[:, :, 0] <= 120)
        & (lower_hsv[:, :, 1] >= 30)
        & (lower_hsv[:, :, 1] <= 130)
        & (lower_hsv[:, :, 2] >= 120)
    )
    ground_structure_ratio = float(1.0 - np.count_nonzero(lower_sky) / lower_sky.size)

    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180.0,
        threshold=max(40, width // 20),
        minLineLength=max(80, width // 10),
        maxLineGap=20,
    )
    horizontal_angles: list[float] = []
    if lines is not None:
        for x1, y1, x2, y2 in lines.reshape(-1, 4):
            angle = math.degrees(math.atan2(float(y2 - y1), float(x2 - x1)))
            normalized = ((angle + 90.0) % 180.0) - 90.0
            if abs(normalized) <= 20.0:
                horizontal_angles.append(normalized)
    horizon_angle_degrees = (
        float(np.median(horizontal_angles)) if horizontal_angles else float("nan")
    )

    checks = {
        "not_blank": bool(brightness_standard_deviation >= 18.0),
        "not_almost_uniform": bool(dominant_color_ratio <= 0.75),
        "sufficient_edges": bool(edge_density >= 0.010),
        "not_mostly_sky": bool(sky_ratio <= 0.72),
        "plausible_ground_region": bool(ground_structure_ratio >= 0.35),
        "horizon_detected": bool(horizontal_angles),
        "horizon_level": bool(horizontal_angles and abs(horizon_angle_degrees) <= 6.0),
    }
    return {
        "width": width,
        "height": height,
        "brightness_standard_deviation": brightness_standard_deviation,
        "dominant_color_ratio": dominant_color_ratio,
        "edge_density": edge_density,
        "sky_ratio": sky_ratio,
        "ground_structure_ratio": ground_structure_ratio,
        "horizon_angle_degrees": horizon_angle_degrees,
        "horizontal_line_count": len(horizontal_angles),
        "checks": checks,
        "passed": all(checks.values()),
    }


def mean_difference(first: np.ndarray, second: np.ndarray) -> float:
    resized = cv2.resize(second, (first.shape[1], first.shape[0]))
    return float(cv2.absdiff(first, resized).mean())


def validate(project_root: Path, output_root: Path) -> dict[str, Any]:
    source_root = project_root / "webots" / "logs" / "stage5a3_closed_loop"
    sidecar_root = project_root / "webots" / "logs" / "stage5b3_live_perception"
    output_root.mkdir(parents=True, exist_ok=True)
    images: dict[str, np.ndarray] = {}
    metrics: dict[str, Any] = {}

    for output_name, source_name in SOURCE_IMAGES.items():
        source_path = source_root / source_name
        image = cv2.imread(str(source_path), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f"Required viewport export is missing: {source_path}")
        destination = output_root / output_name
        if not cv2.imwrite(str(destination), image):
            raise RuntimeError(f"Could not write visual evidence: {destination}")
        images[output_name] = image
        metrics[output_name] = image_metrics(image)

    annotated_candidates = sorted((sidecar_root / "annotated_frames").glob("*.png"))
    if not annotated_candidates:
        raise FileNotFoundError("No annotated Stage 5B frame is available.")
    selected_annotated = output_root / "selected_cv_annotated_frame.png"
    shutil.copy2(annotated_candidates[len(annotated_candidates) // 2], selected_annotated)

    sequence_names = list(SOURCE_IMAGES)
    consecutive_differences = [
        mean_difference(images[first], images[second])
        for first, second in zip(sequence_names, sequence_names[1:], strict=False)
    ]
    left_right_difference = mean_difference(
        images["left_turn_first_person.png"],
        images["right_turn_first_person.png"],
    )
    sequence_checks = {
        "viewpoint_changes_during_navigation": bool(min(consecutive_differences) >= 4.0),
        "left_and_right_turns_differ": bool(left_right_difference >= 8.0),
    }

    perception_summary = json.loads(
        (sidecar_root / "perception_summary.json").read_text(encoding="utf-8")
    )
    stage5b_report = json.loads(
        (
            project_root / "reports" / "perception" / "stage5b" / "stage5b3_runtime_validation.json"
        ).read_text(encoding="utf-8")
    )
    passed = (
        all(item["passed"] for item in metrics.values())
        and all(sequence_checks.values())
        and bool(stage5b_report["runtime_verified"])
        and bool(stage5b_report["cuda_inference_verified"])
    )
    summary = {
        "schema_version": 1,
        "stage": "5B3",
        "runtime_verified": bool(stage5b_report["runtime_verified"]),
        "visual_validation_passed": passed,
        "cv_model_connected": bool(perception_summary["cv_model_connected"]),
        "cuda_inference_verified": bool(perception_summary["cuda_inference_verified"]),
        "inference_count": int(perception_summary["inference_count"]),
        "failure_count": int(perception_summary["failure_count"]),
        "annotated_frame_count": int(perception_summary["annotated_frame_count"]),
        "model_sha256": str(perception_summary["model_sha256"]),
        "policy_controls_motors": bool(stage5b_report["policy_controls_motors"]),
        "motor_motion_source": str(stage5b_report["motor_motion_source"]),
        "image_metrics": metrics,
        "consecutive_view_differences": consecutive_differences,
        "left_right_view_difference": left_right_difference,
        "sequence_checks": sequence_checks,
        "selected_cv_source": annotated_candidates[len(annotated_candidates) // 2].name,
    }
    summary_path = output_root / "stage5b_runtime_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    if not passed:
        raise RuntimeError(f"Stage 5B visual validation failed: {summary_path}")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    summary = validate(arguments.project_root.resolve(), arguments.output.resolve())
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
