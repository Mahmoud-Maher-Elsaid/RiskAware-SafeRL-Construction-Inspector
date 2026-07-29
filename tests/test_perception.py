from __future__ import annotations

import numpy as np
import pytest

from riskaware_saferrl.perception import (
    DeterministicMockPerception,
    DisabledPerception,
    create_perception_backend,
    preprocess_frames,
    validate_class_mapping,
)


def test_disabled_backend_reports_truthful_status() -> None:
    result = DisabledPerception().infer([np.zeros((12, 16, 4), dtype=np.uint8)])
    assert not result.model_connected
    assert result.detections == ((),)


def test_mock_backend_is_deterministic_and_serializable() -> None:
    frame = np.full((20, 30, 3), 127, dtype=np.uint8)
    backend = DeterministicMockPerception(("Person", "Hardhat"))
    first = backend.infer([frame]).to_dict()
    second = backend.infer([frame]).to_dict()
    assert first["detections"] == second["detections"]
    assert first["model_connected"] is False


def test_preprocessing_supports_bgra_batches() -> None:
    tensor = preprocess_frames([np.zeros((10, 20, 4), dtype=np.uint8)] * 2, (32, 18))
    assert tensor.shape == (2, 3, 18, 32)
    assert tensor.dtype.is_floating_point


def test_class_mapping_rejects_unknown_and_duplicate_classes() -> None:
    with pytest.raises(ValueError, match="duplicate"):
        validate_class_mapping(("Person", "Person"))
    with pytest.raises(ValueError, match="Unsupported"):
        validate_class_mapping(("Excavator",))


def test_missing_real_model_is_not_silently_mocked(tmp_path) -> None:
    with pytest.raises(FileNotFoundError):
        create_perception_backend(
            "torchscript", model_path=tmp_path / "missing.pt", class_names=("Person",)
        )
