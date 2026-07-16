import numpy as np

from riskaware_saferrl.envs import ConstructionInspectionEnv
from riskaware_saferrl.safety import (
    CounterfactualLagrangianReward,
    LagrangeMultiplier,
    SafetyShield,
)
from riskaware_saferrl.scenarios import Scenario


def create_scenario(
    *,
    scenario_id: str,
    workers: tuple[tuple[int, int], ...] = (),
    restricted_zones: tuple[tuple[int, int], ...] = (),
    max_steps: int = 20,
) -> Scenario:
    return Scenario(
        scenario_id=scenario_id,
        split="test",
        grid_size=6,
        agent_start=(2, 2),
        obstacles=(),
        hazards=((0, 0),),
        workers=workers,
        restricted_zones=restricted_zones,
        max_steps=max_steps,
        vision_radius=3,
    )


def create_wrapped_environment(
    scenario: Scenario,
    *,
    multiplier_value: float,
) -> CounterfactualLagrangianReward:
    multiplier = LagrangeMultiplier(
        value=multiplier_value,
        learning_rate=0.1,
        maximum=100.0,
    )
    return CounterfactualLagrangianReward(
        SafetyShield(
            ConstructionInspectionEnv(
                scenario=scenario,
            )
        ),
        multiplier,
    )


def test_counterfactual_cost_survives_shield_replacement() -> None:
    environment = create_wrapped_environment(
        create_scenario(
            scenario_id="counterfactual_worker",
            workers=((2, 4),),
        ),
        multiplier_value=2.0,
    )
    environment.reset(seed=0)

    _, reward, _, _, info = environment.step(3)

    assert info["shield_active"] is True
    assert info["cost"] == 0.0
    assert info["proposed_action_cost"] == 1.0
    assert info["proposed_cost_worker"] == 1.0
    assert info["lagrangian_penalty"] == 2.0
    assert reward == info["task_reward"] - 2.0


def test_combined_proposal_cost_counts_each_violation() -> None:
    environment = create_wrapped_environment(
        create_scenario(
            scenario_id="combined_cost",
            workers=((2, 4),),
            restricted_zones=((2, 3),),
        ),
        multiplier_value=1.0,
    )
    environment.reset(seed=0)

    _, _, _, _, info = environment.step(3)

    assert info["cost"] == 0.0
    assert info["proposed_action_cost"] == 2.0
    assert info["proposed_cost_worker"] == 1.0
    assert info["proposed_cost_restricted"] == 1.0


def test_episode_proposed_cost_is_reported() -> None:
    environment = create_wrapped_environment(
        create_scenario(
            scenario_id="episode_cost",
            workers=((2, 4),),
            max_steps=1,
        ),
        multiplier_value=1.0,
    )
    environment.reset(seed=0)

    _, _, terminated, truncated, info = environment.step(3)

    assert not terminated
    assert truncated
    assert info["episode_proposed_action_cost"] == 1.0


def test_counterfactual_wrapper_exposes_masks() -> None:
    scenario = create_scenario(
        scenario_id="counterfactual_masks",
    )
    base_environment = ConstructionInspectionEnv(
        scenario=scenario,
    )
    shield = SafetyShield(base_environment)
    environment = CounterfactualLagrangianReward(
        shield,
        LagrangeMultiplier(),
    )
    environment.reset(seed=0)

    assert np.array_equal(
        environment.action_masks(),
        base_environment.action_masks(),
    )


def test_multiplier_is_bounded_by_maximum() -> None:
    multiplier = LagrangeMultiplier(
        value=0.0,
        learning_rate=1.0,
        maximum=2.0,
    )

    value = multiplier.update(
        observed_cost=100.0,
        cost_limit=0.0,
    )

    assert value == 2.0
