from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest


def load_training_module():
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "train_curriculum_ppo.py"
    spec = importlib.util.spec_from_file_location(
        "train_curriculum_ppo",
        script_path,
    )

    if spec is None or spec.loader is None:
        raise RuntimeError("Could not load the training script.")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def parse_training_args(
    monkeypatch: pytest.MonkeyPatch,
    *arguments: str,
):
    module = load_training_module()
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train_curriculum_ppo.py",
            *arguments,
        ],
    )
    return module, module.parse_args()


def test_twenty_update_calibration_is_valid(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module, args = parse_training_args(
        monkeypatch,
        "--algorithm",
        "maskable_ppo",
        "--shield",
        "--lagrangian",
        "--calibration-run",
        "--updates",
        "20",
        "--easy-updates",
        "5",
        "--medium-updates",
        "5",
    )

    module.validate_args(args)


def test_short_full_run_requires_calibration_flag(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module, args = parse_training_args(
        monkeypatch,
        "--algorithm",
        "maskable_ppo",
        "--shield",
        "--lagrangian",
        "--updates",
        "20",
        "--easy-updates",
        "5",
        "--medium-updates",
        "5",
    )

    with pytest.raises(
        ValueError,
        match="Full curriculum training requires",
    ):
        module.validate_args(args)


def test_smoke_and_calibration_are_mutually_exclusive(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module, args = parse_training_args(
        monkeypatch,
        "--algorithm",
        "maskable_ppo",
        "--shield",
        "--lagrangian",
        "--smoke-test",
        "--calibration-run",
    )

    with pytest.raises(
        ValueError,
        match="cannot be used together",
    ):
        module.validate_args(args)
