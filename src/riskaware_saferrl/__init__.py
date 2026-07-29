"""RiskAware SafeRL package."""

import gymnasium as gym
from gymnasium.envs.registration import register

ENV_ID = "RiskAwareConstruction-v0"

if ENV_ID not in gym.registry:
    register(
        id=ENV_ID,
        entry_point="riskaware_saferrl.envs:ConstructionInspectionEnv",
        max_episode_steps=250,
    )
