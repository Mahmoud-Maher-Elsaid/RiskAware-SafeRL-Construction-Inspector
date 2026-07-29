from __future__ import annotations

import hashlib
import json
import math
import pathlib
import time
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

EXPECTED_PPE_CLASSES = (
    "Fall-Detected",
    "Gloves",
    "Goggles",
    "Hardhat",
    "Ladder",
    "Mask",
    "NO-Gloves",
    "NO-Goggles",
    "NO-Hardhat",
    "NO-Mask",
    "NO-Safety Vest",
    "Person",
    "Safety Cone",
    "Safety Vest",
)

VIOLATION_CLASSES = frozenset(
    {
        "fall-detected",
        "no-gloves",
        "no-goggles",
        "no-hardhat",
        "no-mask",
        "no-safety vest",
    }
)

COMPLIANCE_CLASSES = frozenset(
    {
        "gloves",
        "goggles",
        "hardhat",
        "mask",
        "safety vest",
    }
)

CONTEXT_CLASSES = frozenset(
    {
        "person",
        "ladder",
        "safety cone",
    }
)


def normalize_class_name(value: str) -> str:
    return " ".join(value.strip().lower().replace("_", " ").split())


def sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        while True:
            block = handle.read(1024 * 1024)

            if not block:
                break

            digest.update(block)

    return digest.hexdigest()


def ensure_bgr_frame(frame: np.ndarray) -> np.ndarray:
    array = np.asarray(frame)

    if array.ndim != 3:
        raise ValueError(f"Expected a three-dimensional frame. Received {array.shape}.")

    if array.shape[2] not in {3, 4}:
        raise ValueError("Expected a BGR or BGRA frame with three or four channels.")

    if array.dtype != np.uint8:
        array = np.clip(array, 0, 255).astype(np.uint8)

    if array.shape[2] == 4:
        array = array[:, :, :3]

    return np.ascontiguousarray(array)


@dataclass(frozen=True)
class LiveDetection:
    class_id: int
    class_name: str
    normalized_class_name: str
    confidence: float
    xyxy: tuple[float, float, float, float]

    def to_dict(self) -> dict[str, object]:
        return {
            "class_id": self.class_id,
            "class_name": self.class_name,
            "normalized_class_name": self.normalized_class_name,
            "confidence": self.confidence,
            "xyxy": list(self.xyxy),
        }


@dataclass(frozen=True)
class PerceptionRiskSummary:
    risk_level: str
    person_count: int
    violation_count: int
    compliance_count: int
    context_count: int
    highest_confidence: float
    fall_detected: bool
    ppe_violation_detected: bool
    class_counts: tuple[tuple[str, int], ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "risk_level": self.risk_level,
            "person_count": self.person_count,
            "violation_count": self.violation_count,
            "compliance_count": self.compliance_count,
            "context_count": self.context_count,
            "highest_confidence": self.highest_confidence,
            "fall_detected": self.fall_detected,
            "ppe_violation_detected": self.ppe_violation_detected,
            "class_counts": dict(self.class_counts),
        }


@dataclass(frozen=True)
class LivePerceptionResult:
    frame_id: int
    model_connected: bool
    backend: str
    device: str
    model_sha256: str
    frame_width: int
    frame_height: int
    inference_ms: float
    detections: tuple[LiveDetection, ...]
    risk: PerceptionRiskSummary
    error: str | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "frame_id": self.frame_id,
            "model_connected": self.model_connected,
            "backend": self.backend,
            "device": self.device,
            "model_sha256": self.model_sha256,
            "frame_width": self.frame_width,
            "frame_height": self.frame_height,
            "inference_ms": self.inference_ms,
            "detections": [detection.to_dict() for detection in self.detections],
            "risk": self.risk.to_dict(),
            "error": self.error,
        }


def build_risk_summary(
    detections: Sequence[LiveDetection],
) -> PerceptionRiskSummary:
    class_counter: Counter[str] = Counter(
        detection.normalized_class_name for detection in detections
    )

    violation_count = sum(
        count for class_name, count in class_counter.items() if class_name in VIOLATION_CLASSES
    )

    compliance_count = sum(
        count for class_name, count in class_counter.items() if class_name in COMPLIANCE_CLASSES
    )

    context_count = sum(
        count for class_name, count in class_counter.items() if class_name in CONTEXT_CLASSES
    )

    fall_detected = class_counter.get("fall-detected", 0) > 0
    ppe_violation_detected = any(
        class_name.startswith("no-") and count > 0 for class_name, count in class_counter.items()
    )

    if fall_detected:
        risk_level = "critical"
    elif ppe_violation_detected:
        risk_level = "high"
    elif violation_count > 0:
        risk_level = "high"
    elif context_count > 0:
        risk_level = "observed"
    else:
        risk_level = "none_observed"

    highest_confidence = max(
        (detection.confidence for detection in detections),
        default=0.0,
    )

    return PerceptionRiskSummary(
        risk_level=risk_level,
        person_count=class_counter.get("person", 0),
        violation_count=violation_count,
        compliance_count=compliance_count,
        context_count=context_count,
        highest_confidence=highest_confidence,
        fall_detected=fall_detected,
        ppe_violation_detected=ppe_violation_detected,
        class_counts=tuple(sorted(class_counter.items())),
    )


def validate_class_schema(
    class_names: Sequence[str],
    expected_classes: Sequence[str] = EXPECTED_PPE_CLASSES,
) -> None:
    normalized_actual = [normalize_class_name(name) for name in class_names]

    normalized_expected = [normalize_class_name(name) for name in expected_classes]

    if len(normalized_actual) != len(set(normalized_actual)):
        raise ValueError("The model class mapping contains duplicates.")

    actual_set = set(normalized_actual)
    expected_set = set(normalized_expected)

    if actual_set == expected_set:
        return

    missing = sorted(expected_set - actual_set)
    unexpected = sorted(actual_set - expected_set)

    raise ValueError(
        "The model class mapping does not match the PPE schema. "
        f"Missing={missing}; unexpected={unexpected}."
    )


def load_live_perception_config(
    config_path: pathlib.Path,
) -> dict[str, Any]:
    payload = json.loads(config_path.read_text(encoding="utf-8-sig"))

    if payload.get("backend") != "ultralytics_yolo":
        raise ValueError("Stage 5B requires the ultralytics_yolo backend.")

    if not bool(payload.get("enabled")):
        raise ValueError("Stage 5B live perception is disabled.")

    return payload


class UltralyticsPpeBackend:
    backend_name = "ultralytics_yolo"

    def __init__(
        self,
        *,
        model_path: pathlib.Path,
        expected_sha256: str,
        expected_classes: Sequence[str],
        confidence_threshold: float = 0.25,
        image_size: int = 640,
        device: str = "cuda:0",
        require_cuda: bool = True,
    ) -> None:
        self.model_path = pathlib.Path(model_path).resolve()

        if not self.model_path.exists():
            raise FileNotFoundError(f"PPE model was not found: {self.model_path}")

        actual_sha256 = sha256_file(self.model_path)

        if actual_sha256.lower() != expected_sha256.lower():
            raise ValueError("The PPE model SHA256 does not match the configured artifact.")

        if not 0.0 < confidence_threshold <= 1.0:
            raise ValueError("confidence_threshold must be in the interval (0, 1].")

        if image_size < 32:
            raise ValueError("image_size must be at least 32.")

        import torch
        from ultralytics import YOLO

        if require_cuda and not torch.cuda.is_available():
            raise RuntimeError("CUDA is required for Stage 5B live perception.")

        if device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError(f"The configured CUDA device is unavailable: {device}")

        self._torch = torch
        self._model = YOLO(str(self.model_path))
        self.confidence_threshold = float(confidence_threshold)
        self.image_size = int(image_size)
        self.device = device
        self.model_sha256 = actual_sha256
        self.model_connected = True

        names_payload = self._model.names

        if isinstance(names_payload, list):
            self.class_names = {index: str(name) for index, name in enumerate(names_payload)}
        else:
            self.class_names = {
                int(index): str(name) for index, name in dict(names_payload).items()
            }

        validate_class_schema(
            tuple(self.class_names.values()),
            expected_classes,
        )

    def warmup(
        self,
        frame_shape: tuple[int, int, int] = (360, 640, 3),
    ) -> float:
        frame = np.zeros(frame_shape, dtype=np.uint8)
        started_at = time.perf_counter()

        self._model.predict(
            source=frame,
            imgsz=self.image_size,
            conf=self.confidence_threshold,
            device=self.device,
            verbose=False,
            save=False,
        )

        if self.device.startswith("cuda"):
            self._torch.cuda.synchronize()

        return (time.perf_counter() - started_at) * 1000.0

    def infer(
        self,
        frame: np.ndarray,
        *,
        frame_id: int,
    ) -> LivePerceptionResult:
        bgr_frame = ensure_bgr_frame(frame)
        frame_height, frame_width = bgr_frame.shape[:2]

        started_at = time.perf_counter()

        predictions = self._model.predict(
            source=bgr_frame,
            imgsz=self.image_size,
            conf=self.confidence_threshold,
            device=self.device,
            verbose=False,
            save=False,
        )

        if self.device.startswith("cuda"):
            self._torch.cuda.synchronize()

        inference_ms = (time.perf_counter() - started_at) * 1000.0

        if len(predictions) != 1:
            raise RuntimeError("Expected exactly one Ultralytics prediction result.")

        boxes = predictions[0].boxes
        detections: list[LiveDetection] = []

        if boxes is not None and len(boxes) > 0:
            coordinates = boxes.xyxy.detach().cpu().numpy()
            confidences = boxes.conf.detach().cpu().numpy()
            class_ids = boxes.cls.detach().cpu().numpy().astype(int)

            for xyxy, confidence, class_id in zip(
                coordinates,
                confidences,
                class_ids,
                strict=True,
            ):
                numeric_confidence = float(confidence)

                if not math.isfinite(numeric_confidence):
                    continue

                class_name = self.class_names[int(class_id)]

                detections.append(
                    LiveDetection(
                        class_id=int(class_id),
                        class_name=class_name,
                        normalized_class_name=(normalize_class_name(class_name)),
                        confidence=numeric_confidence,
                        xyxy=tuple(float(value) for value in xyxy.tolist()),
                    )
                )

        risk = build_risk_summary(detections)

        return LivePerceptionResult(
            frame_id=int(frame_id),
            model_connected=True,
            backend=self.backend_name,
            device=self.device,
            model_sha256=self.model_sha256,
            frame_width=frame_width,
            frame_height=frame_height,
            inference_ms=inference_ms,
            detections=tuple(detections),
            risk=risk,
        )


def create_live_perception_backend(
    *,
    project_root: pathlib.Path,
    config_path: pathlib.Path,
) -> UltralyticsPpeBackend:
    config = load_live_perception_config(config_path)
    model_config = config["model"]
    runtime_config = config["runtime"]

    model_path = pathlib.Path(project_root) / model_config["relative_path"]

    return UltralyticsPpeBackend(
        model_path=model_path,
        expected_sha256=model_config["sha256"],
        expected_classes=model_config["expected_classes"],
        confidence_threshold=float(runtime_config["confidence_threshold"]),
        image_size=int(runtime_config["image_size"]),
        device=str(runtime_config["device"]),
        require_cuda=bool(runtime_config["require_cuda"]),
    )


def annotate_frame(
    frame: np.ndarray,
    result: LivePerceptionResult,
) -> np.ndarray:
    import cv2

    annotated = ensure_bgr_frame(frame).copy()

    for detection in result.detections:
        x1, y1, x2, y2 = (int(round(value)) for value in detection.xyxy)

        cv2.rectangle(
            annotated,
            (x1, y1),
            (x2, y2),
            (0, 255, 255),
            2,
        )

        label = f"{detection.class_name} {detection.confidence:.2f}"

        cv2.putText(
            annotated,
            label,
            (max(0, x1), max(15, y1 - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 255, 255),
            1,
            cv2.LINE_AA,
        )

    status_text = (
        f"Stage 5B | {result.device} | {result.inference_ms:.1f} ms | risk={result.risk.risk_level}"
    )

    cv2.putText(
        annotated,
        status_text,
        (10, 22),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )

    return annotated
