import gymnasium as gym
from stable_baselines3.common.env_checker import check_env

import riskaware_saferrl  # noqa: F401
from riskaware_saferrl.envs import ConstructionInspectionEnv


def main() -> None:
    env = ConstructionInspectionEnv()
    check_env(env, warn=True)

    observation, info = env.reset(seed=42)
    assert env.observation_space.contains(observation)
    assert "cost" in info

    for _ in range(20):
        action = env.action_space.sample()
        observation, _, terminated, truncated, info = env.step(action)
        assert env.observation_space.contains(observation)
        assert info["cost"] >= 0.0
        if terminated or truncated:
            observation, info = env.reset()

    registered_env = gym.make(riskaware_saferrl.ENV_ID)
    registered_env.reset(seed=42)
    registered_env.close()
    env.close()

    print("Environment check passed.")


if __name__ == "__main__":
    main()