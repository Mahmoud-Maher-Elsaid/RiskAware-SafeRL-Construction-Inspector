from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

from riskaware_saferrl.envs import (
    ConstructionInspectionEnv,
)
from riskaware_saferrl.safety import (
    CounterfactualLagrangianReward,
    LagrangeMultiplier,
)
from riskaware_saferrl.scenarios import Scenario


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


def test_lagrangian_training_allows_no_shield(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = load_training_module()
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train_curriculum_ppo.py",
            "--algorithm",
            "maskable_ppo",
            "--lagrangian",
            "--calibration-run",
            "--updates",
            "20",
            "--easy-updates",
            "5",
            "--medium-updates",
            "5",
        ],
    )
    args = module.parse_args()

    module.validate_args(args)


def test_unshielded_lagrangian_preserves_transition_alignment() -> None:
    scenario = Scenario(
        scenario_id="intrinsic_lagrangian",
        split="test",
        grid_size=6,
        agent_start=(2, 2),
        obstacles=(),
        hazards=((0, 0),),
        workers=((2, 4),),
        restricted_zones=(),
        max_steps=20,
        vision_radius=3,
    )
    environment = CounterfactualLagrangianReward(
        ConstructionInspectionEnv(
            scenario=scenario,
        ),
        LagrangeMultiplier(
            value=2.0,
            learning_rate=0.01,
            maximum=100.0,
        ),
    )
    environment.reset(seed=0)

    _, reward, _, _, info = environment.step(3)

    assert info["proposed_action_cost"] == 1.0
    assert info["cost"] > 0.0
    assert info["lagrangian_penalty"] == 2.0
    assert reward == info["task_reward"] - 2.0
