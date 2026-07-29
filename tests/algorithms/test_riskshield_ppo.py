from __future__ import annotations

import numpy as np

from riskaware_saferrl.algorithms import RiskShieldCostWrapper
from riskaware_saferrl.envs import GridEnvironmentConfig, ResearchConstructionEnv


def make_wrapper() -> RiskShieldCostWrapper:
    environment = ResearchConstructionEnv(
        GridEnvironmentConfig(
            size=8,
            obstacle_density=0,
            hazard_density=0.02,
            worker_density=0,
            restricted_density=0,
            dynamic_hazard_density=0,
            ppe_risk_density=0,
            dynamic_obstacles=False,
            max_steps=2,
        )
    )
    return RiskShieldCostWrapper(
        environment, safety_budget=0.0, lagrange_learning_rate=0.5, device="cpu"
    )


def test_cost_value_network_updates() -> None:
    wrapper = make_wrapper()
    wrapper.reset(seed=1)
    wrapper.unwrapped.agent = (0, 0)
    wrapper.unwrapped.obstacles = {(0, 1)}
    before = [parameter.detach().clone() for parameter in wrapper.cost_value.parameters()]
    _, _, _, _, info = wrapper.step(3)
    assert info["cost_value_loss"] >= 0
    assert any(
        not np.array_equal(first.numpy(), second.detach().numpy())
        for first, second in zip(before, wrapper.cost_value.parameters(), strict=True)
    )


def test_multiplier_changes_penalized_policy_reward() -> None:
    wrapper = make_wrapper()
    wrapper.reset(seed=1)
    wrapper.unwrapped.agent = (0, 0)
    wrapper.unwrapped.obstacles = {(0, 1)}
    wrapper.step(3)
    _, _, _, _, terminal_info = wrapper.step(3)
    assert terminal_info["lagrangian_multiplier_updated"] > 0
    wrapper.reset(seed=1)
    wrapper.unwrapped.agent = (0, 0)
    wrapper.unwrapped.obstacles = {(0, 1)}
    _, penalized, _, _, info = wrapper.step(3)
    assert penalized < info["raw_reward"]


def test_budget_and_multiplier_telemetry_are_exposed() -> None:
    wrapper = make_wrapper()
    wrapper.reset(seed=3)
    _, _, _, _, info = wrapper.step(4)
    assert info["safety_budget"] == 0.0
    assert "constraint_excess" in info
    assert "lagrangian_multiplier" in info
