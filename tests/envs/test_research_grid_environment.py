from __future__ import annotations

import json

import numpy as np
from gymnasium.utils.env_checker import check_env

from riskaware_saferrl.envs import GridEnvironmentConfig, ResearchConstructionEnv


def make_env(**overrides) -> ResearchConstructionEnv:
    values = {
        "size": 8,
        "obstacle_density": 0.08,
        "hazard_density": 0.04,
        "worker_density": 0.02,
        "restricted_density": 0.03,
        "dynamic_hazard_density": 0.02,
        "ppe_risk_density": 0.02,
        "max_steps": 40,
    }
    values.update(overrides)
    return ResearchConstructionEnv(GridEnvironmentConfig(**values))


def test_gymnasium_checker_and_spaces() -> None:
    environment = make_env()
    check_env(environment)
    observation, _ = environment.reset(seed=7)
    assert environment.observation_space.contains(observation)
    assert environment.action_space.n == 5


def test_seeded_reset_and_step_are_deterministic() -> None:
    first = make_env()
    second = make_env()
    first_observation, first_info = first.reset(seed=19)
    second_observation, second_info = second.reset(seed=19)
    assert first_info == second_info
    for key in first_observation:
        np.testing.assert_array_equal(first_observation[key], second_observation[key])
    action = int(np.flatnonzero(first.action_masks())[0])
    first_transition = first.step(action)
    second_transition = second.step(action)
    for key in first_transition[0]:
        np.testing.assert_array_equal(first_transition[0][key], second_transition[0][key])
    assert first_transition[1:] == second_transition[1:]


def test_partial_observation_and_semantic_channels() -> None:
    environment = make_env(vision_radius=1)
    observation, _ = environment.reset(seed=3)
    visible = observation["map"][9]
    assert 1 <= int(visible.sum()) <= 5
    assert observation["map"].shape[0] == len(environment.CHANNEL_NAMES)
    assert observation["state"][2] == environment.orientation / 3.0


def test_reward_cost_collision_restricted_and_near_miss() -> None:
    environment = make_env(dynamic_obstacles=False)
    environment.reset(seed=1)
    environment.agent = (0, 0)
    environment.obstacles = {(0, 1)}
    environment.workers = set()
    environment.dynamic_hazards = set()
    environment.restricted = set()
    environment.ppe_risk = set()
    _, reward, _, _, info = environment.step(3)
    assert reward < 0
    assert info["cost"] == 1.0
    assert info["collisions"] == 1
    environment.restricted = {(1, 0)}
    environment.workers = {(2, 0)}
    _, _, _, _, info = environment.step(1)
    assert info["restricted_violations"] == 1
    assert info["near_misses"] == 1
    assert info["cost"] == 2.0


def test_hazard_inspection_and_action_mask() -> None:
    environment = make_env(dynamic_obstacles=False)
    environment.reset(seed=2)
    environment.agent = (3, 3)
    environment.hazards = {(3, 4)}
    environment.inspected = set()
    assert environment.action_masks()[4]
    _, reward, terminated, _, info = environment.step(4)
    assert reward > 0
    assert terminated
    assert info["hazard_recall"] == 1.0


def test_perception_noise_preserves_ground_truth() -> None:
    environment = make_env(perception_false_negative_rate=1.0)
    observation, _ = environment.reset(seed=5)
    assert environment.hazards
    assert not observation["map"][1].any()
    assert environment.hazards


def test_serialization_and_rgb_render() -> None:
    environment = ResearchConstructionEnv(GridEnvironmentConfig(size=8), render_mode="rgb_array")
    environment.reset(seed=11)
    serialized = environment.serialize_state()
    json.dumps(serialized)
    image = environment.render()
    assert isinstance(image, np.ndarray)
    assert image.shape == (128, 128, 3)
