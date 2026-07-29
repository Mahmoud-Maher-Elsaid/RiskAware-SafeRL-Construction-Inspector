from __future__ import annotations

import importlib.util
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
VALIDATOR_PATH = PROJECT_ROOT / "scripts" / "validate_stage5b_visual_evidence.py"


def load_validator():
    specification = importlib.util.spec_from_file_location("stage5b_visual", VALIDATOR_PATH)
    if specification is None or specification.loader is None:
        raise RuntimeError("Could not load the Stage 5B visual validator.")
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def test_uniform_sky_image_is_rejected() -> None:
    module = load_validator()
    image = np.full((360, 640, 3), (220, 190, 160), dtype=np.uint8)
    metrics = module.image_metrics(image)
    assert not metrics["passed"]
    assert not metrics["checks"]["not_almost_uniform"]


def test_structured_level_scene_is_accepted() -> None:
    module = load_validator()
    image = np.full((360, 640, 3), (220, 190, 160), dtype=np.uint8)
    image[170:, :] = (80, 85, 90)
    cv2.line(image, (0, 175), (639, 175), (20, 20, 20), 8)
    for x in range(20, 640, 60):
        cv2.rectangle(image, (x, 130), (x + 15, 350), (30, 30, 30), -1)
    metrics = module.image_metrics(image)
    assert metrics["checks"]["horizon_level"]
    assert metrics["checks"]["plausible_ground_region"]
    assert metrics["passed"]
