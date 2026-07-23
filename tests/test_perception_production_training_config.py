from __future__ import annotations

import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

CONFIG_PATH = PROJECT_ROOT / "configs" / "perception" / "production_candidate_100e.json"


def load_config() -> dict:
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


def test_training_uses_exactly_one_hundred_epochs() -> None:
    config = load_config()
    training = config["training"]

    assert training["epochs"] == 100
    assert training["patience"] >= 100
    assert training["fraction"] == 1.0


def test_training_uses_yolo26s_generic_weight() -> None:
    config = load_config()
    model = config["model"]

    assert model["name"] == "yolo26s"

    assert model["generic_weights"] == ("artifacts/models/perception/generic_pretrained/yolo26s.pt")

    assert model["expected_sha256"] == (
        "646f8bc3fe0a656803d95c294f7852321748cb29d13466a1af8862e2db384a1b"
    )


def test_training_never_uses_old_ppe_checkpoints() -> None:
    config = load_config()
    policy = config["training_policy"]

    assert policy["resume"] is False
    assert policy["allow_pilot_checkpoints"] is False

    assert policy["allow_historical_custom_ppe_checkpoint"] is False

    assert policy["allow_official_generic_pretrained_weights"] is True


def test_test_split_is_reserved() -> None:
    config = load_config()
    policy = config["training_policy"]

    assert policy["test_split_allowed"] is False

    assert policy["final_production_model_selected_by_this_stage"] is False


def test_training_uses_robust_augmentation_schedule() -> None:
    config = load_config()
    training = config["training"]

    assert training["image_size"] == 640
    assert training["amp"] is True
    assert training["cosine_learning_rate"] is True
    assert training["multi_scale"] == 0.25
    assert training["mosaic"] == 1.0
    assert training["close_mosaic"] == 15
    assert training["class_weight_power"] == 0.35


def test_long_run_saves_intermediate_checkpoints() -> None:
    config = load_config()
    training = config["training"]

    assert training["save"] is True
    assert training["save_period"] == 5
