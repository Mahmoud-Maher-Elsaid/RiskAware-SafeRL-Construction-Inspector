from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from riskaware_saferrl.live_perception import LiveDetection

VIOLATION_CLASSES = {
    "fall detected",
    "no gloves",
    "no goggles",
    "no hardhat",
    "no mask",
    "no safety vest",
}
WORKER_CLASSES = {"person"}
CONTEXT_CLASSES = {"ladder", "safety cone"}
COMPLIANCE_CLASSES = {"gloves", "goggles", "hardhat", "mask", "safety vest"}


@dataclass(frozen=True)
class ProjectedDetection:
    class_name: str
    confidence: float
    local_x_m: float
    local_z_m: float
    source: str = "cv"


@dataclass(frozen=True)
class SemanticRiskMapUpdate:
    risk_map: np.ndarray
    worker_map: np.ndarray
    ppe_violation_map: np.ndarray
    context_map: np.ndarray
    maximum_risk: float
    action_mask_changed: bool
    emergency_stop: bool
    source: str
    frame_id: int


class TemporalDetectionFilter:
    """Confidence smoothing and duplicate suppression across live frames."""

    def __init__(self, *, alpha: float = 0.6, stale_after_frames: int = 3) -> None:
        if not 0.0 < alpha <= 1.0:
            raise ValueError("alpha must be in (0, 1]")
        if stale_after_frames < 1:
            raise ValueError("stale_after_frames must be positive")
        self.alpha = alpha
        self.stale_after_frames = stale_after_frames
        self._tracks: dict[tuple[str, int, int], tuple[LiveDetection, int]] = {}

    @staticmethod
    def _key(detection: LiveDetection) -> tuple[str, int, int]:
        x1, y1, x2, y2 = detection.xyxy
        return (
            detection.normalized_class_name,
            int(round((x1 + x2) / 40.0)),
            int(round((y1 + y2) / 40.0)),
        )

    def update(
        self, detections: tuple[LiveDetection, ...], frame_id: int
    ) -> tuple[LiveDetection, ...]:
        deduplicated: dict[tuple[str, int, int], LiveDetection] = {}
        for detection in detections:
            key = self._key(detection)
            current = deduplicated.get(key)
            if current is None or detection.confidence > current.confidence:
                deduplicated[key] = detection
        for key, detection in deduplicated.items():
            previous = self._tracks.get(key)
            confidence = detection.confidence
            if previous is not None:
                confidence = (
                    self.alpha * detection.confidence + (1.0 - self.alpha) * previous[0].confidence
                )
            self._tracks[key] = (
                LiveDetection(
                    class_id=detection.class_id,
                    class_name=detection.class_name,
                    normalized_class_name=detection.normalized_class_name,
                    confidence=float(confidence),
                    xyxy=detection.xyxy,
                ),
                frame_id,
            )
        self._tracks = {
            key: value
            for key, value in self._tracks.items()
            if frame_id - value[1] <= self.stale_after_frames
        }
        return tuple(
            value[0] for _, value in sorted(self._tracks.items(), key=lambda item: item[0])
        )


class SemanticRiskMapper:
    """Project live CV detections into a robot-local semantic risk map."""

    def __init__(
        self,
        *,
        map_size: int = 16,
        resolution_m: float = 0.5,
        horizontal_fov_radians: float = 1.05,
        confidence_threshold: float = 0.25,
        emergency_risk_threshold: float = 0.9,
    ) -> None:
        self.map_size = map_size
        self.resolution_m = resolution_m
        self.horizontal_fov_radians = horizontal_fov_radians
        self.confidence_threshold = confidence_threshold
        self.emergency_risk_threshold = emergency_risk_threshold

    def project(
        self, detection: LiveDetection, frame_width: int, frame_height: int
    ) -> ProjectedDetection:
        x1, _, x2, y2 = detection.xyxy
        center_x = (x1 + x2) / 2.0
        normalized_x = center_x / frame_width - 0.5
        angle = normalized_x * self.horizontal_fov_radians
        bottom_fraction = min(1.0, max(0.05, y2 / frame_height))
        forward_distance = max(0.5, 7.0 * (1.0 - bottom_fraction))
        lateral_distance = float(np.tan(angle) * forward_distance)
        return ProjectedDetection(
            class_name=detection.normalized_class_name,
            confidence=detection.confidence,
            local_x_m=lateral_distance,
            local_z_m=forward_distance,
        )

    def update(
        self,
        detections: tuple[LiveDetection, ...],
        *,
        frame_id: int,
        frame_width: int,
        frame_height: int,
    ) -> SemanticRiskMapUpdate:
        risk = np.zeros((self.map_size, self.map_size), dtype=np.float32)
        workers = np.zeros_like(risk)
        violations = np.zeros_like(risk)
        context = np.zeros_like(risk)
        center = self.map_size // 2
        for detection in detections:
            if detection.confidence < self.confidence_threshold:
                continue
            projected = self.project(detection, frame_width, frame_height)
            column = int(round(center + projected.local_x_m / self.resolution_m))
            row = int(round(self.map_size - 1 - projected.local_z_m / self.resolution_m))
            if not (0 <= row < self.map_size and 0 <= column < self.map_size):
                continue
            name = detection.normalized_class_name
            if name in WORKER_CLASSES:
                workers[row, column] = max(workers[row, column], detection.confidence)
                risk[row, column] = max(risk[row, column], 0.85 * detection.confidence)
            elif name in VIOLATION_CLASSES:
                violations[row, column] = max(violations[row, column], detection.confidence)
                risk[row, column] = max(risk[row, column], detection.confidence)
            elif name in CONTEXT_CLASSES:
                context[row, column] = max(context[row, column], detection.confidence)
                risk[row, column] = max(risk[row, column], 0.5 * detection.confidence)
            elif name in COMPLIANCE_CLASSES:
                context[row, column] = max(context[row, column], detection.confidence)
        maximum_risk = float(risk.max())
        return SemanticRiskMapUpdate(
            risk_map=risk,
            worker_map=workers,
            ppe_violation_map=violations,
            context_map=context,
            maximum_risk=maximum_risk,
            action_mask_changed=maximum_risk >= 0.6,
            emergency_stop=maximum_risk >= self.emergency_risk_threshold,
            source="cv",
            frame_id=frame_id,
        )
