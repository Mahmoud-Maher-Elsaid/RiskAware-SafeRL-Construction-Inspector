import numpy as np

from riskaware_saferrl.envs import ConstructionInspectionEnv


def test_environment_obeys_gymnasium_api() -> None:
    env = ConstructionInspectionEnv()
    observation, info = env.reset(seed=7)

    assert env.observation_space.contains(observation)
    assert observation["map"].ndim == 1
    assert observation["map"].shape == (7 * env.size * env.size,)
    assert info["cost"] == 0.0

    observation, reward, terminated, truncated, info = env.step(4)

    assert env.observation_space.contains(observation)
    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    assert info["cost"] >= 0.0


def test_seeded_reset_is_reproducible() -> None:
    first = ConstructionInspectionEnv()
    second = ConstructionInspectionEnv()

    first_observation, _ = first.reset(seed=99)
    second_observation, _ = second.reset(seed=99)

    np.testing.assert_array_equal(
        first_observation["map"],
        second_observation["map"],
    )
    np.testing.assert_array_equal(
        first_observation["state"],
        second_observation["state"],
    )


def test_flattened_semantic_map_can_be_restored() -> None:
    env = ConstructionInspectionEnv(size=12)
    observation, _ = env.reset(seed=5)

    semantic_grid = env.semantic_grid_from_observation(observation)

    assert semantic_grid.shape == (7, 12, 12)
    assert semantic_grid.dtype == np.float32
    assert np.all((semantic_grid >= 0.0) & (semantic_grid <= 1.0))