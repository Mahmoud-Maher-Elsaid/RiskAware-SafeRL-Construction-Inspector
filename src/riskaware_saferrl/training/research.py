from __future__ import annotations

import json
from dataclasses import fields
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
import yaml
from gymnasium import spaces
from stable_baselines3.common.callbacks import BaseCallback

from riskaware_saferrl.envs import GridEnvironmentConfig, ResearchConstructionEnv


def load_grid_config(path: Path) -> GridEnvironmentConfig:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    allowed = {field.name for field in fields(GridEnvironmentConfig)}
    return GridEnvironmentConfig(**{key: value for key, value in payload.items() if key in allowed})


class ContinuousActionAdapter(gym.ActionWrapper):
    """Map SAC's continuous scalar to one of five documented discrete actions."""

    def __init__(self, environment: gym.Env) -> None:
        super().__init__(environment)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)

    def action(self, action: np.ndarray) -> int:
        scalar = float(np.clip(np.asarray(action).reshape(-1)[0], -1.0, 1.0))
        return int(np.clip(np.floor((scalar + 1.0) * 2.5), 0, 4))

    @staticmethod
    def discrete_to_continuous(action: int) -> np.ndarray:
        if not 0 <= action <= 4:
            raise ValueError("Discrete action must be in [0, 4]")
        return np.array([-0.8, -0.4, 0.0, 0.4, 0.8], dtype=np.float32)[[action]]


def flatten_research_environment(
    config: GridEnvironmentConfig,
    *,
    continuous_actions: bool = False,
) -> gym.Env:
    environment: gym.Env = ResearchConstructionEnv(config)
    if continuous_actions:
        environment = ContinuousActionAdapter(environment)
    return gym.wrappers.FlattenObservation(environment)


class EpisodeMetricsCallback(BaseCallback):
    """Persist genuine rollout metrics as interruption-safe JSONL records."""

    def __init__(
        self,
        path: Path,
        *,
        early_stopping_patience_episodes: int | None = None,
        minimum_episodes: int = 20,
        verbose: int = 0,
    ) -> None:
        super().__init__(verbose)
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.episode_reward = np.zeros(1, dtype=np.float64)
        self.episode_cost = np.zeros(1, dtype=np.float64)
        self.episode_steps = np.zeros(1, dtype=np.int64)
        self.episode_index = 0
        self.early_stopping_patience_episodes = early_stopping_patience_episodes
        self.minimum_episodes = minimum_episodes
        self.best_safety_adjusted_return = -float("inf")
        self.episodes_without_improvement = 0

    def _on_training_start(self) -> None:
        count = int(self.training_env.num_envs)
        self.episode_reward = np.zeros(count, dtype=np.float64)
        self.episode_cost = np.zeros(count, dtype=np.float64)
        self.episode_steps = np.zeros(count, dtype=np.int64)

    def _on_step(self) -> bool:
        rewards = np.asarray(self.locals["rewards"], dtype=np.float64)
        dones = np.asarray(self.locals["dones"], dtype=np.bool_)
        infos: list[dict[str, Any]] = self.locals["infos"]
        self.episode_reward += rewards
        self.episode_steps += 1
        self.episode_cost += np.asarray(
            [float(info.get("safety_cost", info.get("cost", 0.0))) for info in infos]
        )
        records: list[dict[str, Any]] = []
        for index, done in enumerate(dones):
            if not done:
                continue
            self.episode_index += 1
            info = infos[index]
            records.append(
                {
                    "episode": self.episode_index,
                    "timesteps": int(self.num_timesteps),
                    "episodic_return": float(self.episode_reward[index]),
                    "safety_cost": float(self.episode_cost[index]),
                    "episode_length": int(self.episode_steps[index]),
                    "coverage": float(info.get("coverage", 0.0)),
                    "hazard_recall": float(info.get("hazard_recall", 0.0)),
                    "collisions": int(info.get("collisions", 0)),
                    "constraint_violations": int(info.get("near_misses", 0))
                    + int(info.get("restricted_violations", 0)),
                    "success": bool(info.get("success", False)),
                }
            )
            safety_adjusted_return = float(self.episode_reward[index] - self.episode_cost[index])
            if safety_adjusted_return > self.best_safety_adjusted_return + 1e-6:
                self.best_safety_adjusted_return = safety_adjusted_return
                self.episodes_without_improvement = 0
            else:
                self.episodes_without_improvement += 1
            self.episode_reward[index] = 0.0
            self.episode_cost[index] = 0.0
            self.episode_steps[index] = 0
        if records:
            with self.path.open("a", encoding="utf-8") as stream:
                for record in records:
                    stream.write(json.dumps(record, sort_keys=True) + "\n")
        if (
            self.early_stopping_patience_episodes is not None
            and self.episode_index >= self.minimum_episodes
            and self.episodes_without_improvement >= self.early_stopping_patience_episodes
        ):
            if self.verbose:
                print(
                    "Early stopping: safety-adjusted episodic return did not improve "
                    f"for {self.episodes_without_improvement} episodes."
                )
            return False
        return True
