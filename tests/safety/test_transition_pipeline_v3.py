from __future__ import annotations

from pathlib import Path

import pytest

from riskaware_saferrl.envs import GridEnvironmentConfig, ResearchConstructionEnvV2
from riskaware_saferrl.safety import (
    ControlledInspectionState,
    SafetyContractV3,
    SafetyContractV3Wrapper,
    SafetyStepContext,
    SafetyVectorCost,
)


def wrapped() -> SafetyContractV3Wrapper:
    environment = ResearchConstructionEnvV2(
        GridEnvironmentConfig(
            size=8,
            obstacle_density=0.0,
            hazard_density=0.02,
            worker_density=0.0,
            restricted_density=0.0,
            dynamic_hazard_density=0.0,
            ppe_risk_density=0.0,
            dynamic_obstacles=False,
            max_steps=2,
        )
    )
    contract = SafetyContractV3.from_yaml(Path("configs/safety/safety_contract_v3.yaml"))
    return SafetyContractV3Wrapper(environment, contract)


def context(action: int) -> SafetyStepContext:
    return SafetyStepContext(
        proposed_action=action,
        predicted_trajectory=((0, 0),),
        predicted_vector_cost=SafetyVectorCost(),
        shield_result={"shield_decision": "accept"},
        inspection=ControlledInspectionState(),
    )


def test_pipeline_attributes_cost_to_executed_action() -> None:
    environment = wrapped()
    observation, _ = environment.reset(seed=3)
    assert observation["state"].shape == (22,)
    assert not observation["state"][17:].any()
    action = int(next(index for index, valid in enumerate(observation["action_mask"]) if valid))
    environment.prepare_step(context(action))
    _, _, _, _, info = environment.step(action)
    record = info["safety_transition_v3"]
    assert record.policy_action == action
    assert record.executed_action == action
    assert info["safety_vector_v3"] == record.actual_vector_cost.as_dict()
    assert record.next_state["state"].shape == (22,)


def test_pipeline_requires_context_and_rejects_post_termination() -> None:
    environment = wrapped()
    observation, _ = environment.reset(seed=4)
    action = int(next(index for index, valid in enumerate(observation["action_mask"]) if valid))
    with pytest.raises(RuntimeError, match="context"):
        environment.step(action)
    for _ in range(2):
        environment.prepare_step(context(action))
        _, _, terminated, truncated, _ = environment.step(action)
        if terminated or truncated:
            break
    assert terminated or truncated
    with pytest.raises(RuntimeError, match="post-termination"):
        environment.prepare_step(context(action))
