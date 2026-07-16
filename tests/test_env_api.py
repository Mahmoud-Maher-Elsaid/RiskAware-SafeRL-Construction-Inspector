import numpy as np

from riskaware_saferrl.envs import ConstructionInspectionEnv


def test_environment_obeys_gymnasium_api() -> None:
    env = ConstructionInspectionEnv()
    observation, info = env.reset(seed=7)

    assert env.observation_space.contains(observation)
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

    np.testing.assert_array_equal(first_observation["map"], second_observation["map"])
    np.testing.assert_array_equal(first_observation["state"], second_observation["state"])