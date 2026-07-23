from __future__ import annotations

import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "configs" / "perception" / "pilot_training.json"


def load_config() -> dict:
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


def test_pilot_training_uses_fresh_experiments() -> None:
    config = load_config()
    policy = config["training_policy"]

    assert policy["resume"] is False
    assert policy["allow_historical_custom_ppe_checkpoint"] is False
    assert policy["allow_official_generic_pretrained_weights"] is True
    assert policy["overwrite_existing_runs"] is False
    assert policy["test_split_allowed"] is False


def test_pilot_candidates_use_only_approved_generic_weights() -> None:
    config = load_config()
    candidates = config["candidates"]

    assert [candidate["name"] for candidate in candidates] == [
        "yolo26n",
        "yolo26s",
    ]

    assert [candidate["weights"] for candidate in candidates] == [
        "yolo26n.pt",
        "yolo26s.pt",
    ]

    for candidate in candidates:
        weight_path = Path(candidate["weights"])

        assert weight_path.name == candidate["weights"]
        assert not weight_path.is_absolute()


def test_pilot_does_not_use_test_split() -> None:
    config = load_config()

    assert config["training_policy"]["test_split_allowed"] is False
    assert config["subset"]["train_images"] == 1400
    assert config["subset"]["validation_images"] == 500


def test_pilot_training_is_bounded() -> None:
    config = load_config()
    training = config["training"]

    assert training["epochs"] == 1
    assert training["image_size"] == 640
    assert training["device"] == 0
    assert training["amp"] is True
    assert training["cache"] is False
    assert training["validation"] is True
