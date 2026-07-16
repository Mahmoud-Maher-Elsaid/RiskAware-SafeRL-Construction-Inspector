from typing import Any

import numpy as np

from riskaware_saferrl.evaluation.scenario_evaluator import (
    evaluate_policy_on_scenarios,
)
from riskaware_saferrl.scenarios import Scenario


class RecordingPolicy:
    def __init__(self) -> None:
        self.received_masks: list[np.ndarray | None] = []

    def predict(
        self,
        observation: dict[str, np.ndarray],
        *,
        deterministic: bool = True,
        action_masks: np.ndarray | None = None,
        **kwargs: Any,
    ) -> tuple[np.ndarray, None]:
        del observation, deterministic, kwargs
        self.received_masks.append(action_masks)
        return np.asarray(4), None


def create_evaluation_scenario() -> Scenario:
    return Scenario(
        scenario_id="masked_eval",
        split="validation",
        grid_size=6,
        agent_start=(0, 0),
        obstacles=(),
        hazards=((0, 1),),
        workers=(),
        restricted_zones=(),
        max_steps=5,
        vision_radius=3,
    )


def test_masked_evaluation_passes_current_action_masks() -> None:
    policy = RecordingPolicy()

    _, summary = evaluate_policy_on_scenarios(
        policy,
        [create_evaluation_scenario()],
        use_action_masks=True,
    )

    assert policy.received_masks
    assert policy.received_masks[0] is not None
    assert bool(policy.received_masks[0][4])
    assert summary["action_masking"] is True
    assert summary["metrics"]["success"]["mean"] == 1.0


def test_unmasked_evaluation_preserves_existing_behavior() -> None:
    policy = RecordingPolicy()

    _, summary = evaluate_policy_on_scenarios(
        policy,
        [create_evaluation_scenario()],
        use_action_masks=False,
    )

    assert policy.received_masks == [None]
    assert summary["action_masking"] is False
