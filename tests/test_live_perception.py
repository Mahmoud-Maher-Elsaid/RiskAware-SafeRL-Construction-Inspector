from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from riskaware_saferrl.live_perception import (
    EXPECTED_PPE_CLASSES,
    LiveDetection,
    UltralyticsPpeBackend,
    build_risk_summary,
    ensure_bgr_frame,
    normalize_class_name,
    validate_class_schema,
)


def test_normalize_class_name_is_stable() -> None:
    assert normalize_class_name("NO_Safety   Vest") == "no safety vest"


def test_webots_bgra_frame_becomes_bgr() -> None:
    frame = np.zeros((12, 20, 4), dtype=np.uint8)
    frame[:, :, 0] = 10
    frame[:, :, 1] = 20
    frame[:, :, 2] = 30
    frame[:, :, 3] = 255

    converted = ensure_bgr_frame(frame)

    assert converted.shape == (12, 20, 3)
    assert converted.dtype == np.uint8
    assert converted[0, 0].tolist() == [10, 20, 30]


def test_risk_summary_marks_ppe_violation() -> None:
    detections = (
        LiveDetection(
            class_id=11,
            class_name="Person",
            normalized_class_name="person",
            confidence=0.91,
            xyxy=(1.0, 2.0, 10.0, 20.0),
        ),
        LiveDetection(
            class_id=8,
            class_name="NO-Hardhat",
            normalized_class_name="no-hardhat",
            confidence=0.82,
            xyxy=(2.0, 3.0, 9.0, 11.0),
        ),
    )

    summary = build_risk_summary(detections)

    assert summary.risk_level == "high"
    assert summary.person_count == 1
    assert summary.violation_count == 1
    assert summary.ppe_violation_detected


def test_class_schema_accepts_expected_ppe_classes() -> None:
    validate_class_schema(EXPECTED_PPE_CLASSES)


def test_class_schema_rejects_unknown_class() -> None:
    invalid_classes = list(EXPECTED_PPE_CLASSES)
    invalid_classes[-1] = "Excavator"

    with pytest.raises(ValueError, match="does not match"):
        validate_class_schema(invalid_classes)


def test_missing_model_fails_truthfully(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        UltralyticsPpeBackend(
            model_path=tmp_path / "missing.pt",
            expected_sha256="0" * 64,
            expected_classes=EXPECTED_PPE_CLASSES,
        )


def test_stage5b_config_is_portable() -> None:
    project_root = Path(__file__).resolve().parents[1]
    config_path = project_root / "configs" / "perception" / "stage5b_live_perception.json"

    payload = json.loads(config_path.read_text(encoding="utf-8-sig"))

    relative_path = payload["model"]["relative_path"]

    assert not Path(relative_path).is_absolute()
    assert payload["backend"] == "ultralytics_yolo"
    assert payload["runtime"]["require_cuda"] is True
    assert payload["safety"]["policy_controls_motors"] is False
    assert len(payload["model"]["sha256"]) == 64
