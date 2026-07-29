from __future__ import annotations

import numpy as np

from riskaware_saferrl.webots.bridge import GridAction
from riskaware_saferrl.webots.runtime_control import (
    RuntimeSafetyEvaluator,
    inject_perception_risk,
)


def test_safety_evaluator_changes_unsafe_policy_action() -> None:
    shield = RuntimeSafetyEvaluator()
    shield.configure(
        forbidden_actions={int(GridAction.MOVE_RIGHT)},
        emergency_stop=False,
    )
    executed, decision, cost = shield.evaluate(int(GridAction.MOVE_RIGHT))
    assert executed == int(GridAction.INSPECT)
    assert decision == "safe_projection"
    assert cost > 0.0
    assert shield.intervention_count == 1


def test_emergency_stop_blocks_motion() -> None:
    shield = RuntimeSafetyEvaluator()
    shield.configure(forbidden_actions=set(), emergency_stop=True)
    executed, decision, _ = shield.evaluate(int(GridAction.MOVE_UP))
    assert executed == int(GridAction.INSPECT)
    assert decision == "emergency_hold"


def test_cv_risk_changes_checkpoint_compatible_observation() -> None:
    original = {
        "map": np.zeros(7 * 12 * 12, dtype=np.float32),
        "state": np.zeros(4, dtype=np.float32),
    }
    updated = inject_perception_risk(
        original,
        agent_position=(4, 5),
        grid_size=12,
        risk_score=0.85,
    )
    assert updated["map"].shape == (1008,)
    assert updated["state"].shape == (4,)
    assert updated["map"].reshape(7, 12, 12)[6, 4, 5] == np.float32(0.85)
    assert not np.array_equal(updated["map"], original["map"])
