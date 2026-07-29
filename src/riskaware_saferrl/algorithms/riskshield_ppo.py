from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np
import torch
from torch import nn

from riskaware_saferrl.safety.lagrangian import LagrangeMultiplier


class CostValueNetwork(nn.Module):
    """Auxiliary state-cost value estimator for PPO-Lagrangian."""

    def __init__(self, input_dim: int, hidden_dim: int = 128) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        return self.network(observation)


def flatten_observation(observation: dict[str, np.ndarray]) -> np.ndarray:
    return np.concatenate(
        [np.asarray(observation[key], dtype=np.float32).reshape(-1) for key in sorted(observation)]
    )


class RiskShieldCostWrapper(gym.Wrapper):
    """Apply PPO-Lagrangian reward shaping with a learned cost value function."""

    def __init__(
        self,
        environment: gym.Env,
        *,
        safety_budget: float = 5.0,
        lagrange_learning_rate: float = 0.02,
        cost_learning_rate: float = 1e-3,
        gamma: float = 0.99,
        device: str = "cuda",
    ) -> None:
        super().__init__(environment)
        self.safety_budget = safety_budget
        self.gamma = gamma
        self.device = torch.device(
            device if device != "cuda" or torch.cuda.is_available() else "cpu"
        )
        sample, _ = self.env.reset(seed=0)
        input_dim = flatten_observation(sample).size
        self.cost_value = CostValueNetwork(input_dim).to(self.device)
        self.cost_optimizer = torch.optim.Adam(self.cost_value.parameters(), lr=cost_learning_rate)
        self.lagrange = LagrangeMultiplier(
            value=0.0, learning_rate=lagrange_learning_rate, maximum=100.0
        )
        self.previous_observation = sample
        self.episode_cost = 0.0
        self.last_cost_value_loss = 0.0

    def reset(self, **kwargs):
        observation, info = self.env.reset(**kwargs)
        self.previous_observation = observation
        self.episode_cost = 0.0
        return observation, info

    def _update_cost_value(
        self,
        observation: dict[str, np.ndarray],
        next_observation: dict[str, np.ndarray],
        cost: float,
        done: bool,
    ) -> float:
        state = torch.as_tensor(flatten_observation(observation), device=self.device).unsqueeze(0)
        next_state = torch.as_tensor(
            flatten_observation(next_observation), device=self.device
        ).unsqueeze(0)
        predicted = self.cost_value(state).squeeze(-1)
        with torch.no_grad():
            target = torch.tensor([cost], dtype=torch.float32, device=self.device)
            if not done:
                target = target + self.gamma * self.cost_value(next_state).squeeze(-1)
        loss = nn.functional.mse_loss(predicted, target)
        self.cost_optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.cost_value.parameters(), 1.0)
        self.cost_optimizer.step()
        return float(loss.detach().cpu())

    def step(self, action: int):
        observation, raw_reward, terminated, truncated, info = self.env.step(action)
        cost = float(info.get("safety_cost", info.get("cost", 0.0)))
        done = bool(terminated or truncated)
        self.last_cost_value_loss = self._update_cost_value(
            self.previous_observation, observation, cost, done
        )
        multiplier_before = self.lagrange.value
        penalized_reward = self.lagrange.penalized_reward(float(raw_reward), cost)
        self.episode_cost += cost
        if done:
            self.lagrange.update(self.episode_cost, self.safety_budget)
        self.previous_observation = observation
        enriched: dict[str, Any] = dict(info)
        enriched.update(
            {
                "raw_reward": float(raw_reward),
                "penalized_reward": float(penalized_reward),
                "cost_value_loss": self.last_cost_value_loss,
                "lagrangian_multiplier": float(multiplier_before),
                "lagrangian_multiplier_updated": float(self.lagrange.value),
                "safety_budget": float(self.safety_budget),
                "constraint_excess": float(self.episode_cost - self.safety_budget),
            }
        )
        return observation, float(penalized_reward), terminated, truncated, enriched

    def save_cost_value(self, path: str) -> None:
        torch.save(self.cost_value.state_dict(), path)
