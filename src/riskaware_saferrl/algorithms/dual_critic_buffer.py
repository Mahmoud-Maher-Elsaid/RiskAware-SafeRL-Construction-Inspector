from __future__ import annotations

from collections.abc import Generator
from typing import NamedTuple

import numpy as np
import torch
from gymnasium import spaces
from sb3_contrib.common.maskable.buffers import (
    MaskableDictRolloutBuffer,
)
from stable_baselines3.common.type_aliases import TensorDict
from stable_baselines3.common.vec_env import VecNormalize


class DualCriticMaskableDictRolloutBufferSamples(NamedTuple):
    observations: TensorDict
    actions: torch.Tensor
    old_values: torch.Tensor
    old_log_prob: torch.Tensor
    advantages: torch.Tensor
    returns: torch.Tensor
    action_masks: torch.Tensor
    old_cost_values: torch.Tensor
    cost_advantages: torch.Tensor
    cost_returns: torch.Tensor


class DualCriticMaskableDictRolloutBuffer(MaskableDictRolloutBuffer):
    """Maskable rollout buffer with a separate safety-cost GAE."""

    def __init__(
        self,
        buffer_size: int,
        observation_space: spaces.Dict,
        action_space: spaces.Space,
        device: torch.device | str = "auto",
        gae_lambda: float = 1.0,
        gamma: float = 0.99,
        n_envs: int = 1,
        *,
        cost_gamma: float = 1.0,
        cost_gae_lambda: float = 0.95,
    ) -> None:
        if not 0.0 <= cost_gamma <= 1.0:
            raise ValueError("cost_gamma must be in [0, 1].")
        if not 0.0 <= cost_gae_lambda <= 1.0:
            raise ValueError("cost_gae_lambda must be in [0, 1].")

        self.cost_gamma = float(cost_gamma)
        self.cost_gae_lambda = float(cost_gae_lambda)

        super().__init__(
            buffer_size,
            observation_space,
            action_space,
            device,
            gae_lambda,
            gamma,
            n_envs=n_envs,
        )

    def reset(self) -> None:
        self.costs = np.zeros(
            (self.buffer_size, self.n_envs),
            dtype=np.float32,
        )
        self.cost_values = np.zeros(
            (self.buffer_size, self.n_envs),
            dtype=np.float32,
        )
        self.cost_advantages = np.zeros(
            (self.buffer_size, self.n_envs),
            dtype=np.float32,
        )
        self.cost_returns = np.zeros(
            (self.buffer_size, self.n_envs),
            dtype=np.float32,
        )
        super().reset()

    def add(
        self,
        *args,
        costs: np.ndarray,
        cost_values: torch.Tensor,
        action_masks: np.ndarray | None = None,
        **kwargs,
    ) -> None:
        self.costs[self.pos] = np.asarray(
            costs,
            dtype=np.float32,
        ).reshape(self.n_envs)
        self.cost_values[self.pos] = cost_values.detach().cpu().numpy().reshape(self.n_envs)

        super().add(
            *args,
            action_masks=action_masks,
            **kwargs,
        )

    def compute_cost_returns_and_advantage(
        self,
        last_cost_values: torch.Tensor,
        dones: np.ndarray,
    ) -> None:
        last_values = last_cost_values.detach().cpu().numpy().reshape(self.n_envs)
        last_gae = np.zeros(
            self.n_envs,
            dtype=np.float32,
        )

        for step in reversed(range(self.buffer_size)):
            if step == self.buffer_size - 1:
                next_non_terminal = 1.0 - np.asarray(
                    dones,
                    dtype=np.float32,
                )
                next_values = last_values
            else:
                next_non_terminal = 1.0 - self.episode_starts[step + 1]
                next_values = self.cost_values[step + 1]

            delta = (
                self.costs[step]
                + self.cost_gamma * next_values * next_non_terminal
                - self.cost_values[step]
            )
            last_gae = delta + self.cost_gamma * self.cost_gae_lambda * next_non_terminal * last_gae
            self.cost_advantages[step] = last_gae

        self.cost_returns = self.cost_advantages + self.cost_values

    def get(
        self,
        batch_size: int | None = None,
    ) -> Generator[
        DualCriticMaskableDictRolloutBufferSamples,
        None,
        None,
    ]:
        assert self.full

        indices = np.random.permutation(self.buffer_size * self.n_envs)

        if not self.generator_ready:
            for key, observation in self.observations.items():
                self.observations[key] = self.swap_and_flatten(observation)

            tensor_names = (
                "actions",
                "values",
                "log_probs",
                "advantages",
                "returns",
                "action_masks",
                "cost_values",
                "cost_advantages",
                "cost_returns",
            )
            for tensor_name in tensor_names:
                self.__dict__[tensor_name] = self.swap_and_flatten(self.__dict__[tensor_name])

            self.generator_ready = True

        if batch_size is None:
            batch_size = self.buffer_size * self.n_envs

        start_index = 0
        while start_index < self.buffer_size * self.n_envs:
            batch_indices = indices[start_index : start_index + batch_size]
            yield self._get_samples(batch_indices)
            start_index += batch_size

    def _get_samples(
        self,
        batch_indices: np.ndarray,
        env: VecNormalize | None = None,
    ) -> DualCriticMaskableDictRolloutBufferSamples:
        del env

        return DualCriticMaskableDictRolloutBufferSamples(
            observations={
                key: self.to_torch(observation[batch_indices])
                for key, observation in self.observations.items()
            },
            actions=self.to_torch(self.actions[batch_indices]),
            old_values=self.to_torch(self.values[batch_indices].flatten()),
            old_log_prob=self.to_torch(self.log_probs[batch_indices].flatten()),
            advantages=self.to_torch(self.advantages[batch_indices].flatten()),
            returns=self.to_torch(self.returns[batch_indices].flatten()),
            action_masks=self.to_torch(
                self.action_masks[batch_indices].reshape(-1, self.mask_dims)
            ),
            old_cost_values=self.to_torch(self.cost_values[batch_indices].flatten()),
            cost_advantages=self.to_torch(self.cost_advantages[batch_indices].flatten()),
            cost_returns=self.to_torch(self.cost_returns[batch_indices].flatten()),
        )
