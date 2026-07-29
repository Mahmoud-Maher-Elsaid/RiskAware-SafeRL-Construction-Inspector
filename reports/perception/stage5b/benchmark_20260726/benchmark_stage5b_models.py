from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import math
import pathlib
import statistics
import time
from collections import Counter
from typing import Any

import cv2
import numpy as np
import torch
from ultralytics import YOLO

MODEL_PATHS = (
    "artifacts/runs/perception_weak_class_fine_tuning/"
    "yolo26s_weak_class_20e_20260723_151531/weights/best.pt",
    "artifacts/runs/perception_production_optimized_100e/"
    "yolo26s_gpu90_fixed640_100e_seed42_20260723_003324/weights/best.pt",
    "artifacts/runs/perception_fast_balanced/"
    "yolo26s_fast512_35e_seed42_20260723_000121/weights/best.pt",
    "artifacts/runs/perception_production_100e/yolo26s_100e_seed42_20260722_214549/weights/best.pt",
    "artifacts/runs/perception_pilots/yolo26s_20260722_211614_epoch1_batch4/weights/best.pt",
    "artifacts/runs/perception_pilots/yolo26n_20260722_211614_epoch1_batch8/weights/best.pt",
)

VIOLATION_CLASS_NAMES = {
    "fall-detected",
    "no-gloves",
    "no-goggles",
    "no-hardhat",
    "no-mask",
    "no-safety vest",
}

PPE_CLASS_NAMES = {
    "gloves",
    "goggles",
    "hardhat",
    "mask",
    "safety vest",
}

IMPORTANT_CLASS_NAMES = (
    VIOLATION_CLASS_NAMES
    | PPE_CLASS_NAMES
    | {
        "person",
        "ladder",
        "safety cone",
    }
)


def sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        while True:
            block = handle.read(1024 * 1024)

            if not block:
                break

            digest.update(block)

    return digest.hexdigest()


def normalize_class_name(value: str) -> str:
    return " ".join(value.strip().lower().replace("_", " ").split())


def finite_float(value: Any, default: float = 0.0) -> float:
    try:
        numeric_value = float(value)
    except (TypeError, ValueError):
        return default

    if not math.isfinite(numeric_value):
        return default

    return numeric_value


def percentile(values: list[float], quantile: float) -> float:
    if not values:
        return 0.0

    sorted_values = sorted(values)

    if len(sorted_values) == 1:
        return sorted_values[0]

    position = (len(sorted_values) - 1) * quantile
    lower_index = math.floor(position)
    upper_index = math.ceil(position)

    if lower_index == upper_index:
        return sorted_values[lower_index]

    lower_weight = upper_index - position
    upper_weight = position - lower_index

    return sorted_values[lower_index] * lower_weight + sorted_values[upper_index] * upper_weight


def result_detections(
    result: Any,
    class_names: dict[int, str],
) -> list[dict[str, Any]]:
    detections: list[dict[str, Any]] = []

    boxes = getattr(result, "boxes", None)

    if boxes is None or len(boxes) == 0:
        return detections

    xyxy = boxes.xyxy.detach().cpu().numpy()
    confidences = boxes.conf.detach().cpu().numpy()
    class_ids = boxes.cls.detach().cpu().numpy().astype(int)

    for coordinates, confidence, class_id in zip(
        xyxy,
        confidences,
        class_ids,
        strict=True,
    ):
        raw_name = str(class_names.get(int(class_id), class_id))
        normalized_name = normalize_class_name(raw_name)

        detections.append(
            {
                "class_id": int(class_id),
                "class_name": raw_name,
                "normalized_class_name": normalized_name,
                "confidence": finite_float(confidence),
                "xyxy": [finite_float(value) for value in coordinates.tolist()],
            }
        )

    return detections


def load_image(path: pathlib.Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)

    if image is None:
        raise RuntimeError(f"Could not read image: {path}")

    if image.ndim != 3 or image.shape[2] != 3:
        raise RuntimeError(f"Unexpected image shape: {path} -> {image.shape}")

    return image


def benchmark_model(
    *,
    model_path: pathlib.Path,
    image_paths: list[pathlib.Path],
    device: str,
    confidence_threshold: float,
    image_size: int,
) -> dict[str, Any]:
    started_at = time.perf_counter()
    model = YOLO(str(model_path))
    load_seconds = time.perf_counter() - started_at

    names_payload = getattr(model, "names", {})

    if isinstance(names_payload, list):
        class_names = {index: str(name) for index, name in enumerate(names_payload)}
    else:
        class_names = {int(index): str(name) for index, name in dict(names_payload).items()}

    normalized_model_classes = {normalize_class_name(name) for name in class_names.values()}

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    warmup_image = load_image(image_paths[0])

    warmup_start = time.perf_counter()

    model.predict(
        source=warmup_image,
        imgsz=image_size,
        conf=confidence_threshold,
        device=device,
        verbose=False,
        save=False,
    )

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    warmup_seconds = time.perf_counter() - warmup_start

    inference_times_ms: list[float] = []
    total_detections = 0
    frame_detection_counts: list[int] = []
    confidence_values: list[float] = []
    class_counter: Counter[str] = Counter()
    important_counter: Counter[str] = Counter()
    violation_counter: Counter[str] = Counter()
    ppe_counter: Counter[str] = Counter()
    per_frame_records: list[dict[str, Any]] = []

    for image_path in image_paths:
        image = load_image(image_path)

        inference_start = time.perf_counter()

        prediction = model.predict(
            source=image,
            imgsz=image_size,
            conf=confidence_threshold,
            device=device,
            verbose=False,
            save=False,
        )

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        elapsed_ms = (time.perf_counter() - inference_start) * 1000.0
        inference_times_ms.append(elapsed_ms)

        if len(prediction) != 1:
            raise RuntimeError(f"Expected one prediction result for {image_path}.")

        detections = result_detections(
            prediction[0],
            class_names,
        )

        total_detections += len(detections)
        frame_detection_counts.append(len(detections))

        frame_classes: Counter[str] = Counter()

        for detection in detections:
            class_name = detection["normalized_class_name"]
            confidence = detection["confidence"]

            confidence_values.append(confidence)
            class_counter[class_name] += 1
            frame_classes[class_name] += 1

            if class_name in IMPORTANT_CLASS_NAMES:
                important_counter[class_name] += 1

            if class_name in VIOLATION_CLASS_NAMES:
                violation_counter[class_name] += 1

            if class_name in PPE_CLASS_NAMES:
                ppe_counter[class_name] += 1

        per_frame_records.append(
            {
                "image": str(image_path),
                "inference_ms": elapsed_ms,
                "detection_count": len(detections),
                "class_counts": dict(sorted(frame_classes.items())),
                "detections": detections,
            }
        )

    cuda_peak_memory_bytes = 0

    if torch.cuda.is_available():
        cuda_peak_memory_bytes = int(torch.cuda.max_memory_allocated())

    image_count = len(image_paths)
    frames_with_detections = sum(count > 0 for count in frame_detection_counts)
    frames_with_person = sum(
        record["class_counts"].get("person", 0) > 0 for record in per_frame_records
    )
    frames_with_violation = sum(
        any(
            class_name in VIOLATION_CLASS_NAMES and count > 0
            for class_name, count in record["class_counts"].items()
        )
        for record in per_frame_records
    )

    mean_inference_ms = statistics.fmean(inference_times_ms)
    p95_inference_ms = percentile(inference_times_ms, 0.95)
    maximum_inference_ms = max(inference_times_ms)
    theoretical_fps = 1000.0 / mean_inference_ms if mean_inference_ms > 0.0 else 0.0

    mean_confidence = statistics.fmean(confidence_values) if confidence_values else 0.0

    detection_frame_ratio = frames_with_detections / image_count if image_count else 0.0
    person_frame_ratio = frames_with_person / image_count if image_count else 0.0
    violation_frame_ratio = frames_with_violation / image_count if image_count else 0.0

    important_class_diversity = len(important_counter)
    total_class_diversity = len(class_counter)

    latency_score = max(
        0.0,
        1.0 - min(mean_inference_ms, 200.0) / 200.0,
    )

    confidence_score = min(max(mean_confidence, 0.0), 1.0)

    detection_score = min(
        total_detections / max(image_count * 5.0, 1.0),
        1.0,
    )

    diversity_score = min(
        important_class_diversity / 8.0,
        1.0,
    )

    person_score = min(person_frame_ratio, 1.0)
    violation_score = min(violation_frame_ratio, 1.0)

    live_readiness_score = (
        latency_score * 0.30
        + confidence_score * 0.15
        + detection_score * 0.15
        + diversity_score * 0.15
        + person_score * 0.15
        + violation_score * 0.10
    )

    return {
        "model_path": str(model_path),
        "model_sha256": sha256_file(model_path),
        "model_size_bytes": model_path.stat().st_size,
        "model_size_megabytes": round(
            model_path.stat().st_size / (1024 * 1024),
            3,
        ),
        "device": device,
        "confidence_threshold": confidence_threshold,
        "image_size": image_size,
        "load_seconds": load_seconds,
        "warmup_seconds": warmup_seconds,
        "image_count": image_count,
        "total_detections": total_detections,
        "frames_with_detections": frames_with_detections,
        "detection_frame_ratio": detection_frame_ratio,
        "frames_with_person": frames_with_person,
        "person_frame_ratio": person_frame_ratio,
        "frames_with_violation": frames_with_violation,
        "violation_frame_ratio": violation_frame_ratio,
        "mean_detections_per_frame": (
            statistics.fmean(frame_detection_counts) if frame_detection_counts else 0.0
        ),
        "mean_confidence": mean_confidence,
        "minimum_confidence": (min(confidence_values) if confidence_values else 0.0),
        "maximum_confidence": (max(confidence_values) if confidence_values else 0.0),
        "mean_inference_ms": mean_inference_ms,
        "p95_inference_ms": p95_inference_ms,
        "maximum_inference_ms": maximum_inference_ms,
        "theoretical_fps": theoretical_fps,
        "cuda_peak_memory_bytes": cuda_peak_memory_bytes,
        "cuda_peak_memory_megabytes": round(
            cuda_peak_memory_bytes / (1024 * 1024),
            3,
        ),
        "class_count": len(class_names),
        "model_classes": dict(sorted(class_names.items())),
        "normalized_model_classes": sorted(normalized_model_classes),
        "detected_class_diversity": total_class_diversity,
        "important_class_diversity": important_class_diversity,
        "class_counts": dict(sorted(class_counter.items())),
        "important_class_counts": dict(sorted(important_counter.items())),
        "violation_class_counts": dict(sorted(violation_counter.items())),
        "ppe_class_counts": dict(sorted(ppe_counter.items())),
        "live_readiness_score": live_readiness_score,
        "per_frame_records": per_frame_records,
    }


def write_csv(
    path: pathlib.Path,
    rows: list[dict[str, Any]],
) -> None:
    fieldnames = [
        "rank",
        "model_path",
        "model_size_megabytes",
        "device",
        "image_count",
        "total_detections",
        "frames_with_detections",
        "detection_frame_ratio",
        "frames_with_person",
        "person_frame_ratio",
        "frames_with_violation",
        "violation_frame_ratio",
        "mean_detections_per_frame",
        "mean_confidence",
        "mean_inference_ms",
        "p95_inference_ms",
        "maximum_inference_ms",
        "theoretical_fps",
        "cuda_peak_memory_megabytes",
        "detected_class_diversity",
        "important_class_diversity",
        "live_readiness_score",
        "status",
        "error",
    ]

    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fieldnames,
            extrasaction="ignore",
        )
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", required=True)
    parser.add_argument("--evidence-root", required=True)
    parser.add_argument("--json-report", required=True)
    parser.add_argument("--csv-report", required=True)
    parser.add_argument("--selection-report", required=True)
    parser.add_argument("--confidence", type=float, default=0.25)
    parser.add_argument("--image-size", type=int, default=640)
    parser.add_argument("--maximum-images", type=int, default=0)
    args = parser.parse_args()

    repo_root = pathlib.Path(args.repo_root).resolve()
    evidence_root = pathlib.Path(args.evidence_root).resolve()

    image_paths = sorted(evidence_root.glob("*.png"))

    if args.maximum_images > 0:
        image_paths = image_paths[: args.maximum_images]

    if not image_paths:
        raise RuntimeError("No evidence images were found.")

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for Stage 5B1.")

    device = "cuda:0"

    results: list[dict[str, Any]] = []

    for relative_model_path in MODEL_PATHS:
        model_path = repo_root / relative_model_path

        print("")
        print("=" * 72)
        print(f"BENCHMARKING_MODEL={model_path}")
        print("=" * 72)

        if not model_path.exists():
            result = {
                "model_path": str(model_path),
                "status": "missing",
                "error": "Model file was not found.",
                "live_readiness_score": -1.0,
            }
            results.append(result)
            print("MODEL_STATUS=MISSING")
            continue

        try:
            result = benchmark_model(
                model_path=model_path,
                image_paths=image_paths,
                device=device,
                confidence_threshold=args.confidence,
                image_size=args.image_size,
            )
            result["status"] = "passed"
            result["error"] = None
        except Exception as exc:
            result = {
                "model_path": str(model_path),
                "status": "failed",
                "error": f"{type(exc).__name__}: {exc}",
                "live_readiness_score": -1.0,
            }

        results.append(result)

        print(f"MODEL_STATUS={result['status']}")
        print(f"LIVE_READINESS_SCORE={result.get('live_readiness_score', -1.0):.6f}")
        print(f"MEAN_INFERENCE_MS={result.get('mean_inference_ms', 0.0):.3f}")
        print(f"P95_INFERENCE_MS={result.get('p95_inference_ms', 0.0):.3f}")
        print(f"TOTAL_DETECTIONS={result.get('total_detections', 0)}")
        print(f"IMPORTANT_CLASS_DIVERSITY={result.get('important_class_diversity', 0)}")

        del result
        gc.collect()
        torch.cuda.empty_cache()

    passed_results = [result for result in results if result.get("status") == "passed"]

    passed_results.sort(
        key=lambda result: (
            result["live_readiness_score"],
            result["important_class_diversity"],
            result["person_frame_ratio"],
            -result["mean_inference_ms"],
        ),
        reverse=True,
    )

    failed_results = [result for result in results if result.get("status") != "passed"]

    ranked_results = passed_results + failed_results

    for rank, result in enumerate(ranked_results, start=1):
        result["rank"] = rank

    selected_model = passed_results[0] if passed_results else None

    report = {
        "schema_version": 1,
        "stage": "5B1",
        "benchmark_name": "webots_evidence_model_qualification",
        "device": device,
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "cuda_device_name": torch.cuda.get_device_name(0),
        "confidence_threshold": args.confidence,
        "image_size": args.image_size,
        "evidence_root": str(evidence_root),
        "evidence_image_count": len(image_paths),
        "candidate_count": len(MODEL_PATHS),
        "passed_candidate_count": len(passed_results),
        "failed_candidate_count": len(failed_results),
        "selected_model": selected_model,
        "ranked_results": ranked_results,
    }

    json_report = pathlib.Path(args.json_report)
    csv_report = pathlib.Path(args.csv_report)
    selection_report = pathlib.Path(args.selection_report)

    json_report.parent.mkdir(parents=True, exist_ok=True)

    json_report.write_text(
        json.dumps(report, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    write_csv(csv_report, ranked_results)

    selection_payload = {
        "schema_version": 1,
        "stage": "5B1",
        "selection_status": ("selected" if selected_model is not None else "failed"),
        "selected_model_path": (
            selected_model["model_path"] if selected_model is not None else None
        ),
        "selected_model_sha256": (
            selected_model["model_sha256"] if selected_model is not None else None
        ),
        "selected_live_readiness_score": (
            selected_model["live_readiness_score"] if selected_model is not None else None
        ),
        "selected_mean_inference_ms": (
            selected_model["mean_inference_ms"] if selected_model is not None else None
        ),
        "selected_p95_inference_ms": (
            selected_model["p95_inference_ms"] if selected_model is not None else None
        ),
        "selected_theoretical_fps": (
            selected_model["theoretical_fps"] if selected_model is not None else None
        ),
        "selected_total_detections": (
            selected_model["total_detections"] if selected_model is not None else None
        ),
        "selected_class_counts": (
            selected_model["class_counts"] if selected_model is not None else None
        ),
        "benchmark_json": str(json_report),
        "benchmark_csv": str(csv_report),
    }

    selection_report.write_text(
        json.dumps(
            selection_payload,
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    print("")
    print("=" * 72)
    print("STAGE_5B1_RESULT")
    print("=" * 72)
    print(f"PASSED_CANDIDATES={len(passed_results)}")
    print(f"FAILED_CANDIDATES={len(failed_results)}")

    for result in ranked_results:
        print(
            f"RANK_{result['rank']}="
            f"{result['model_path']} | "
            f"status={result['status']} | "
            f"score={result.get('live_readiness_score', -1.0):.6f} | "
            f"mean_ms={result.get('mean_inference_ms', 0.0):.3f} | "
            f"detections={result.get('total_detections', 0)} | "
            f"important_classes="
            f"{result.get('important_class_diversity', 0)}"
        )

    if selected_model is None:
        print("SELECTED_MODEL=NONE")
        return 2

    print(f"SELECTED_MODEL={selected_model['model_path']}")
    print(f"SELECTED_MODEL_SHA256={selected_model['model_sha256']}")
    print(f"SELECTED_LIVE_READINESS_SCORE={selected_model['live_readiness_score']:.6f}")
    print(f"SELECTED_MEAN_INFERENCE_MS={selected_model['mean_inference_ms']:.3f}")
    print(f"SELECTED_P95_INFERENCE_MS={selected_model['p95_inference_ms']:.3f}")
    print(f"SELECTED_THEORETICAL_FPS={selected_model['theoretical_fps']:.3f}")
    print(f"JSON_REPORT={json_report}")
    print(f"CSV_REPORT={csv_report}")
    print(f"SELECTION_REPORT={selection_report}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
