from __future__ import annotations

import numpy as np
from gymnasium.utils.env_checker import check_env

from riskaware_saferrl.envs import (
    GridEnvironmentConfig,
    ResearchConstructionEnvV2,
    RewardCostV2Config,
)


def make_env(**overrides) -> ResearchConstructionEnvV2:
    values = {
        "size": 8,
        "observation_size": 16,
        "obstacle_density": 0.05,
        "hazard_density": 0.04,
        "worker_density": 0.01,
        "restricted_density": 0.02,
        "dynamic_hazard_density": 0.0,
        "ppe_risk_density": 0.01,
        "dynamic_obstacles": False,
        "max_steps": 60,
    }
    values.update(overrides)
    return ResearchConstructionEnvV2(GridEnvironmentConfig(**values))


def test_v2_environment_checker_and_schema() -> None:
    environment = make_env()
    check_env(environment)
    observation, _ = environment.reset(seed=7)
    assert environment.observation_space.contains(observation)
    assert observation["map"].shape == (11, 16, 16)
    assert observation["state"].shape == (17,)
    np.testing.assert_array_equal(observation["action_mask"], environment.action_masks())


def test_perception_evidence_is_stable_within_episode() -> None:
    environment = make_env(perception_false_negative_rate=0.5)
    first, _ = environment.reset(seed=9)
    second = environment._observation()
    np.testing.assert_array_equal(first["map"][1], second["map"][1])
    np.testing.assert_array_equal(first["map"][3], second["map"][3])


def test_reward_and_cost_are_independent() -> None:
    environment = make_env()
    environment.reset(seed=1)
    environment.agent = (3, 3)
    environment.hazards = {(3, 4)}
    environment.inspected = set()
    environment.workers = {(3, 3)}
    _, reward, terminated, _, info = environment.step(4)
    assert terminated
    assert reward > 10.0
    assert info["cost"] > 0.0
    assert info["inspection_coverage"] == 1.0


def test_progress_shaping_is_monotonic_and_loop_has_no_gain() -> None:
    environment = make_env()
    environment.reset(seed=3)
    environment.agent = (4, 1)
    environment.hazards = {(4, 5)}
    environment.inspected = set()
    environment.obstacles = set()
    environment.workers = set()
    environment.restricted = set()
    environment.dynamic_hazards = set()
    environment.ppe_risk = set()
    _, toward_reward, *_ = environment.step(3)
    _, away_reward, *_ = environment.step(2)
    assert toward_reward > away_reward
    assert toward_reward + away_reward < 0.2


def test_repeated_useless_actions_are_penalized() -> None:
    environment = make_env()
    environment.reset(seed=5)
    environment.agent = (3, 3)
    environment.hazards = {(7, 7)}
    first_reward = environment.step(4)[1]
    second_reward = environment.step(4)[1]
    assert second_reward < first_reward


def test_completion_is_largest_terminal_reward() -> None:
    environment = make_env()
    environment.reset(seed=2)
    environment.agent = (3, 3)
    environment.hazards = {(3, 4)}
    environment.inspected = set()
    completion_reward = environment.step(4)[1]
    assert completion_reward >= environment.reward_cost_config.completion_reward


def test_safety_budget_is_compatible_with_avoidable_costs() -> None:
    cfg = RewardCostV2Config()
    environment = make_env()
    environment.reset(seed=11)
    assert cfg.safety_budget > cfg.collision_cost
    assert cfg.safety_budget > cfg.restricted_cost
    assert environment._info(0.0, False)["success"] is False
