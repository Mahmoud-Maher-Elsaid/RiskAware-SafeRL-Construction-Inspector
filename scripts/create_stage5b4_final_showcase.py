from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import cv2
from PIL import Image


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8-sig"))

    if not isinstance(value, dict):
        raise TypeError(f"Expected a JSON object: {path}")

    return value


def save_json(
    path: Path,
    payload: dict[str, Any],
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    path.write_text(
        json.dumps(
            payload,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--annotated-root",
        required=True,
    )
    parser.add_argument(
        "--sidecar-summary",
        required=True,
    )
    parser.add_argument(
        "--runtime-report",
        required=True,
    )
    parser.add_argument(
        "--artifact-root",
        required=True,
    )
    parser.add_argument(
        "--report",
        required=True,
    )

    args = parser.parse_args()

    annotated_root = Path(args.annotated_root).resolve()

    sidecar_summary_path = Path(args.sidecar_summary).resolve()

    runtime_report_path = Path(args.runtime_report).resolve()

    artifact_root = Path(args.artifact_root).resolve()

    report_path = Path(args.report).resolve()

    image_paths = sorted(annotated_root.glob("*.png"))

    if not image_paths:
        raise RuntimeError("No annotated Stage 5B3 frames were found.")

    sidecar_summary = load_json(sidecar_summary_path)

    runtime_report = load_json(runtime_report_path)

    artifact_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    first_frame = cv2.imread(
        str(image_paths[0]),
        cv2.IMREAD_COLOR,
    )

    if first_frame is None:
        raise RuntimeError("Could not read the first annotated frame.")

    height, width = first_frame.shape[:2]

    mp4_path = artifact_root / "stage5b4_final_live_perception.mp4"

    video_writer = cv2.VideoWriter(
        str(mp4_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        2.0,
        (width, height),
    )

    if not video_writer.isOpened():
        raise RuntimeError("Could not create the showcase MP4.")

    try:
        for image_path in image_paths:
            frame = cv2.imread(
                str(image_path),
                cv2.IMREAD_COLOR,
            )

            if frame is None:
                raise RuntimeError(f"Could not read: {image_path}")

            video_writer.write(frame)
    finally:
        video_writer.release()

    gif_path = artifact_root / "stage5b4_final_live_perception.gif"

    images = [Image.open(path).convert("RGB") for path in image_paths]

    images[0].save(
        gif_path,
        save_all=True,
        append_images=images[1:],
        duration=500,
        loop=0,
        optimize=True,
    )

    for image in images:
        image.close()

    risk_counts = Counter(
        {
            str(key): int(value)
            for key, value in sidecar_summary.get(
                "risk_counts",
                {},
            ).items()
        }
    )

    if risk_counts.get("observed", 0) > 0:
        advisory = "inspection_attention_recommended"
    else:
        advisory = "continue_with_detection_uncertainty"

    report = {
        "schema_version": 1,
        "stage": "5B4-final",
        "runtime_verified": bool(
            runtime_report.get(
                "runtime_verified",
                False,
            )
        ),
        "route_completed": bool(
            runtime_report.get(
                "route_completed",
                False,
            )
        ),
        "returned_to_start": bool(
            runtime_report.get(
                "returned_to_start",
                False,
            )
        ),
        "camera_level_mount_enabled": True,
        "camera_dynamic_stabilization_claimed": False,
        "camera_rotation": "0 1 0 -1.5708",
        "camera_field_of_view_radians": 1.05,
        "cv_model_connected": bool(
            sidecar_summary.get(
                "cv_model_connected",
                False,
            )
        ),
        "cuda_inference_verified": bool(
            sidecar_summary.get(
                "cuda_inference_verified",
                False,
            )
        ),
        "inference_count": int(
            sidecar_summary.get(
                "inference_count",
                0,
            )
        ),
        "annotated_frame_count": len(image_paths),
        "total_detections": int(
            sidecar_summary.get(
                "total_detections",
                0,
            )
        ),
        "detected_class_diversity": int(
            sidecar_summary.get(
                "detected_class_diversity",
                0,
            )
        ),
        "class_counts": sidecar_summary.get(
            "class_counts",
            {},
        ),
        "risk_counts": dict(sorted(risk_counts.items())),
        "safety_advisory": advisory,
        "policy_controls_motors": False,
        "perception_can_stop_robot": False,
        "absence_of_detection_is_not_a_safety_claim": True,
        "github_showcase_ready": (
            mp4_path.is_file() and gif_path.is_file() and len(image_paths) >= 20
        ),
        "gif_path": str(gif_path),
        "mp4_path": str(mp4_path),
        "limitations": {
            "collision_free_claimed": False,
            "real_world_safety_claimed": False,
            "distance_sensor_coverage_verified": False,
            "camera_can_follow_transient_chassis_tilt": True,
        },
    }

    save_json(
        report_path,
        report,
    )

    print(
        json.dumps(
            report,
            indent=2,
            sort_keys=True,
        )
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
