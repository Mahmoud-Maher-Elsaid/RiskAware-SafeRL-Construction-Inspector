from __future__ import annotations

import sys
from pathlib import Path

import yaml

from scripts.evaluate_riskshield_hrmppo_v2 import parse_args as parse_evaluation_args
from scripts.train_riskshield_hrmppo_v2 import parse_args as parse_training_args


def test_v2_training_default_matches_declared_predictive_shield_horizon(
    monkeypatch,
) -> None:
    monkeypatch.setattr(sys, "argv", ["evaluate", "--checkpoint", "dummy.pt"])
    evaluation_args = parse_evaluation_args()
    monkeypatch.setattr(sys, "argv", ["train"])
    training_args = parse_training_args()
    config = yaml.safe_load(
        Path("configs/training/riskshield_hrmppo_v2.yaml").read_text(encoding="utf-8")
    )
    declared_horizon = int(config["shield"]["horizon"])
    assert evaluation_args.shield_horizon == declared_horizon
    assert training_args.shield_horizon == declared_horizon
