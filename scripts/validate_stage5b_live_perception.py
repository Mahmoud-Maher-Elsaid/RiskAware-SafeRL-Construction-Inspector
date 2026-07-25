from __future__ import annotations

import argparse
import json
import pathlib
import statistics
from collections import Counter
from typing import Any

import cv2

from riskaware_saferrl.live_perception import (
    annotate_frame,
    create_live_perception_backend,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--evidence-root", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    project_root = pathlib.Path(args.project_root).resolve()
    config_path = pathlib.Path(args.config).resolve()
    evidence_root = pathlib.Path(args.evidence_root).resolve()
    output_root = pathlib.Path(args.output).resolve()

    image_paths = sorted(evidence_root.glob("*.png"))

    if not image_paths:
        raise RuntimeError("No Stage 5A3 evidence frames were found.")

    annotated_root = output_root / "annotated_frames"
    annotated_root.mkdir(parents=True, exist_ok=True)

    backend = create_live_perception_backend(
        project_root=project_root,
        config_path=config_path,
    )

    warmup_ms = backend.warmup()

    frame_records: list[dict[str, Any]] = []
    inference_times: list[float] = []
    class_counter: Counter[str] = Counter()
    total_detections = 0

    for frame_id, image_path in enumerate(image_paths):
        frame = cv2.imread(
            str(image_path),
            cv2.IMREAD_COLOR,
        )

        if frame is None:
            raise RuntimeError(f"Could not read evidence frame: {image_path}")

        result = backend.infer(
            frame,
            frame_id=frame_id,
        )

        if not result.model_connected:
            raise RuntimeError("The real perception backend reported disconnected.")

        inference_times.append(result.inference_ms)
        total_detections += len(result.detections)

        for detection in result.detections:
            class_counter[detection.normalized_class_name] += 1

        annotated = annotate_frame(frame, result)
        annotated_path = annotated_root / image_path.name

        if not cv2.imwrite(str(annotated_path), annotated):
            raise RuntimeError(f"Could not save annotated frame: {annotated_path}")

        frame_record = result.to_dict()
        frame_record["source_image"] = str(image_path)
        frame_record["annotated_image"] = str(annotated_path)
        frame_records.append(frame_record)

    mean_inference_ms = statistics.fmean(inference_times)
    maximum_inference_ms = max(inference_times)

    sorted_times = sorted(inference_times)
    p95_index = max(
        0,
        min(
            len(sorted_times) - 1,
            round((len(sorted_times) - 1) * 0.95),
        ),
    )
    p95_inference_ms = sorted_times[p95_index]

    report = {
        "schema_version": 1,
        "stage": "5B2",
        "runtime_verified": True,
        "cv_model_connected": True,
        "backend": backend.backend_name,
        "device": backend.device,
        "cuda_inference_verified": backend.device.startswith("cuda"),
        "model_path": str(backend.model_path),
        "model_sha256": backend.model_sha256,
        "warmup_ms": warmup_ms,
        "evidence_frame_count": len(image_paths),
        "inference_count": len(frame_records),
        "total_detections": total_detections,
        "detected_class_diversity": len(class_counter),
        "class_counts": dict(sorted(class_counter.items())),
        "mean_inference_ms": mean_inference_ms,
        "p95_inference_ms": p95_inference_ms,
        "maximum_inference_ms": maximum_inference_ms,
        "all_frames_annotated": (len(list(annotated_root.glob("*.png"))) == len(image_paths)),
        "policy_controls_motors": False,
        "absence_of_detection_is_not_a_safety_claim": True,
        "frame_records": frame_records,
    }

    report_path = output_root / "stage5b2_validation_report.json"

    report_path.write_text(
        json.dumps(
            report,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    summary = {key: value for key, value in report.items() if key != "frame_records"}

    print(json.dumps(summary, indent=2, sort_keys=True))
    print(f"VALIDATION_REPORT={report_path}")
    print(f"ANNOTATED_FRAMES={annotated_root}")

    if not report["cv_model_connected"]:
        return 2

    if not report["cuda_inference_verified"]:
        return 3

    if report["inference_count"] != report["evidence_frame_count"]:
        return 4

    if not report["all_frames_annotated"]:
        return 5

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
