from __future__ import annotations

import numpy as np
import pytest

from riskaware_saferrl.webots.bridge import CardinalHeading
from riskaware_saferrl.webots.policy_dry_run import PolicyDryRunEngine
from riskaware_saferrl.webots.safe_policy_pipeline import SafePolicyPipeline


class FakeModel:
    def predict(self, observation, *, deterministic, action_masks):
        del observation, deterministic
        return np.asarray([int(np.flatnonzero(action_masks)[0])]), None


class FakeShield:
    def __init__(self, executed: int, resolution: str, cost: float) -> None:
        self.executed = executed
        self.resolution = resolution
        self.cost = cost

    def evaluate(self, proposed_action: int) -> tuple[int, str, float]:
        assert proposed_action == 1
        return self.executed, self.resolution, self.cost


def observation() -> dict[str, np.ndarray]:
    return {
        "map": np.zeros(1008, dtype=np.float32),
        "state": np.zeros(4, dtype=np.float32),
    }


def test_pipeline_applies_mask_shield_fallback_before_translation() -> None:
    pipeline = SafePolicyPipeline(
        PolicyDryRunEngine(FakeModel()), FakeShield(4, "emergency_hold", 0.0)
    )
    trace = pipeline.execute(
        sample_index=0,
        observation=observation(),
        task_valid_mask=np.asarray([False, True, False, False, True]),
        heading=CardinalHeading.EAST,
    )
    assert trace.proposed_action == 1
    assert trace.executed_action == 4
    assert trace.fallback_decision == "emergency_hold"
    assert trace.motor_commands == ((0.0, 0.0),)
    assert not trace.policy_controls_motors


def test_motor_gate_rejects_unverified_enablement() -> None:
    with pytest.raises(ValueError, match="verified runtime gate"):
        SafePolicyPipeline(
            PolicyDryRunEngine(FakeModel()),
            FakeShield(1, "not_needed", 0.0),
            policy_controls_motors=True,
        )
