from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Protocol

import cv2
import numpy as np
import torch

CONSTRUCTION_SAFETY_CLASSES = (
    "Person",
    "Hardhat",
    "No Hardhat",
    "Safety Vest",
    "No Safety Vest",
    "Gloves",
    "No Gloves",
    "Goggles",
    "No Goggles",
    "Mask",
    "No Mask",
    "Ladder",
    "Safety Cone",
    "Fall Detected",
)


@dataclass(frozen=True)
class Detection:
    class_id: int
    class_name: str
    confidence: float
    xyxy: tuple[float, float, float, float]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class PerceptionBatch:
    backend: str
    device: str
    model_connected: bool
    latency_ms: float
    detections: tuple[tuple[Detection, ...], ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "backend": self.backend,
            "device": self.device,
            "model_connected": self.model_connected,
            "latency_ms": self.latency_ms,
            "detections": [
                [detection.to_dict() for detection in frame] for frame in self.detections
            ],
        }


class PerceptionBackend(Protocol):
    name: str
    device: str
    model_connected: bool

    def infer(self, frames: Sequence[np.ndarray]) -> PerceptionBatch: ...


def validate_class_mapping(class_names: Sequence[str]) -> tuple[str, ...]:
    names = tuple(str(name).strip() for name in class_names)
    if not names or any(not name for name in names):
        raise ValueError("Class mapping must contain non-empty names.")
    if len(set(names)) != len(names):
        raise ValueError("Class mapping contains duplicate names.")
    unknown = set(names) - set(CONSTRUCTION_SAFETY_CLASSES)
    if unknown:
        raise ValueError(f"Unsupported construction-safety classes: {sorted(unknown)}")
    return names


def preprocess_frames(frames: Sequence[np.ndarray], image_size: tuple[int, int]) -> torch.Tensor:
    if not frames:
        raise ValueError("At least one camera frame is required.")
    width, height = image_size
    tensors: list[torch.Tensor] = []
    for frame in frames:
        array = np.asarray(frame)
        if array.ndim != 3 or array.shape[2] not in (3, 4):
            raise ValueError("Frames must be HxWx3 BGR or HxWx4 BGRA arrays.")
        bgr = array[:, :, :3]
        rgb = cv2.cvtColor(cv2.resize(bgr, (width, height)), cv2.COLOR_BGR2RGB)
        tensors.append(torch.from_numpy(rgb.copy()).permute(2, 0, 1).float().div_(255.0))
    return torch.stack(tensors)


class DisabledPerception:
    name = "disabled"
    device = "none"
    model_connected = False

    def infer(self, frames: Sequence[np.ndarray]) -> PerceptionBatch:
        if not frames:
            raise ValueError("At least one camera frame is required.")
        return PerceptionBatch(self.name, self.device, False, 0.0, tuple(() for _ in frames))


class DeterministicMockPerception:
    name = "mock"
    device = "cpu"
    model_connected = False

    def __init__(self, class_names: Sequence[str] = CONSTRUCTION_SAFETY_CLASSES) -> None:
        self.class_names = validate_class_mapping(class_names)

    def infer(self, frames: Sequence[np.ndarray]) -> PerceptionBatch:
        started = time.perf_counter()
        results: list[tuple[Detection, ...]] = []
        for frame in frames:
            array = np.asarray(frame)
            if array.ndim != 3:
                raise ValueError("Mock frames must be color images.")
            height, width = array.shape[:2]
            checksum = int(array.astype(np.uint64).sum() % len(self.class_names))
            confidence = 0.5 + float(array.mean() / 510.0)
            results.append(
                (
                    Detection(
                        checksum,
                        self.class_names[checksum],
                        min(confidence, 0.99),
                        (width * 0.25, height * 0.25, width * 0.75, height * 0.75),
                    ),
                )
            )
        latency = (time.perf_counter() - started) * 1000.0
        return PerceptionBatch(self.name, self.device, False, latency, tuple(results))


class TorchScriptPerception:
    name = "torchscript"
    model_connected = True

    def __init__(
        self,
        model_path: Path,
        class_names: Sequence[str],
        *,
        confidence_threshold: float = 0.4,
        device: str = "auto",
        image_size: tuple[int, int] = (640, 360),
    ) -> None:
        if not model_path.is_file():
            raise FileNotFoundError(f"Perception model artifact is missing: {model_path}")
        if not 0.0 <= confidence_threshold <= 1.0:
            raise ValueError("confidence_threshold must be between zero and one.")
        self.class_names = validate_class_mapping(class_names)
        self.confidence_threshold = confidence_threshold
        self.image_size = image_size
        requested = "cuda" if device == "auto" and torch.cuda.is_available() else device
        if requested == "auto":
            requested = "cpu"
        if requested.startswith("cuda") and not torch.cuda.is_available():
            requested = "cpu"
        self.device = requested
        self.model = torch.jit.load(str(model_path), map_location=self.device).eval()

    def infer(self, frames: Sequence[np.ndarray]) -> PerceptionBatch:
        tensor = preprocess_frames(frames, self.image_size).to(self.device)
        started = time.perf_counter()
        with torch.inference_mode():
            output = self.model(tensor)
        if self.device.startswith("cuda"):
            torch.cuda.synchronize()
        latency = (time.perf_counter() - started) * 1000.0
        if not isinstance(output, torch.Tensor) or output.ndim != 3 or output.shape[2] != 6:
            raise ValueError("TorchScript detector output must have shape [batch, detections, 6].")
        batches: list[tuple[Detection, ...]] = []
        for rows in output.detach().cpu().numpy():
            detections: list[Detection] = []
            for x1, y1, x2, y2, confidence, class_id_value in rows:
                class_id = int(class_id_value)
                if confidence < self.confidence_threshold:
                    continue
                if not 0 <= class_id < len(self.class_names):
                    raise ValueError(f"Detector produced unmapped class id: {class_id}")
                detections.append(
                    Detection(
                        class_id,
                        self.class_names[class_id],
                        float(confidence),
                        (float(x1), float(y1), float(x2), float(y2)),
                    )
                )
            batches.append(tuple(detections))
        return PerceptionBatch(self.name, self.device, True, latency, tuple(batches))


def create_perception_backend(
    backend: str,
    *,
    model_path: Path | None = None,
    class_names: Sequence[str] = CONSTRUCTION_SAFETY_CLASSES,
    confidence_threshold: float = 0.4,
    device: str = "auto",
) -> PerceptionBackend:
    if backend == "disabled":
        return DisabledPerception()
    if backend == "mock":
        return DeterministicMockPerception(class_names)
    if backend == "torchscript":
        if model_path is None:
            raise ValueError("model_path is required for the TorchScript backend.")
        return TorchScriptPerception(
            model_path,
            class_names,
            confidence_threshold=confidence_threshold,
            device=device,
        )
    raise ValueError(f"Unsupported perception backend: {backend}")
