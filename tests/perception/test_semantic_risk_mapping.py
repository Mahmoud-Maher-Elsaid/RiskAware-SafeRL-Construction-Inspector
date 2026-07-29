from __future__ import annotations

import pytest

from riskaware_saferrl.live_perception import LiveDetection
from riskaware_saferrl.perception_mapping import (
    SemanticRiskMapper,
    TemporalDetectionFilter,
)


def detection(name: str, confidence: float = 0.9) -> LiveDetection:
    return LiveDetection(
        class_id=8,
        class_name=name,
        normalized_class_name=name.lower().replace("-", " "),
        confidence=confidence,
        xyxy=(280.0, 180.0, 360.0, 430.0),
    )


def test_violation_changes_semantic_map_mask_and_emergency_state() -> None:
    mapper = SemanticRiskMapper(emergency_risk_threshold=0.85)
    update = mapper.update(
        (detection("NO-Hardhat"),), frame_id=1, frame_width=640, frame_height=480
    )
    assert update.ppe_violation_map.max() == pytest.approx(0.9)
    assert update.maximum_risk == pytest.approx(0.9)
    assert update.action_mask_changed
    assert update.emergency_stop
    assert update.source == "cv"


def test_confidence_threshold_and_class_scope_are_truthful() -> None:
    mapper = SemanticRiskMapper()
    update = mapper.update(
        (detection("Hole", 0.99), detection("Person", 0.1)),
        frame_id=2,
        frame_width=640,
        frame_height=480,
    )
    assert update.maximum_risk == 0.0


def test_temporal_filter_smooths_deduplicates_and_expires() -> None:
    filter_ = TemporalDetectionFilter(alpha=0.5, stale_after_frames=2)
    first = filter_.update((detection("Person", 0.5), detection("Person", 0.7)), 1)
    assert len(first) == 1
    second = filter_.update((detection("Person", 0.9),), 2)
    assert len(second) == 1
    assert second[0].confidence == 0.8
    assert filter_.update((), 5) == ()


def test_projection_is_deterministic_and_local() -> None:
    mapper = SemanticRiskMapper()
    projected = mapper.project(detection("Safety Cone"), 640, 480)
    assert projected.local_x_m == 0.0
    assert 0.5 <= projected.local_z_m <= 7.0
    assert projected.source == "cv"
