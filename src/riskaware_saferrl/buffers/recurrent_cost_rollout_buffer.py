from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass
class RolloutBatch:
    maps: Tensor
    states: Tensor
    recurrent_states: Tensor
    episode_starts: Tensor
    actions: Tensor
    action_masks: Tensor
    rewards: Tensor
    costs: Tensor
    reward_values: Tensor
    cost_values: Tensor
    log_probabilities: Tensor
    reward_advantages: Tensor
    cost_advantages: Tensor
    returns: Tensor
    cost_returns: Tensor


class RecurrentCostRolloutBuffer:
    """Sequence rollout storage with independent reward and cost GAE."""

    def __init__(self, gamma: float = 0.99, gae_lambda: float = 0.95) -> None:
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.records: list[dict[str, Tensor]] = []

    def add(self, **record: Tensor) -> None:
        required = {
            "map",
            "state",
            "recurrent_state",
            "episode_start",
            "action",
            "action_mask",
            "reward",
            "cost",
            "reward_value",
            "cost_value",
            "log_probability",
            "done",
        }
        missing = required - record.keys()
        if missing:
            raise ValueError(f"Missing rollout fields: {sorted(missing)}")
        self.records.append(record)

    @staticmethod
    def _gae(
        signal: Tensor,
        values: Tensor,
        dones: Tensor,
        last_value: Tensor,
        gamma: float,
        gae_lambda: float,
    ) -> tuple[Tensor, Tensor]:
        advantages = torch.zeros_like(signal)
        running = torch.zeros_like(last_value)
        for index in range(len(signal) - 1, -1, -1):
            next_value = last_value if index == len(signal) - 1 else values[index + 1]
            not_done = 1.0 - dones[index].float()
            delta = signal[index] + gamma * next_value * not_done - values[index]
            running = delta + gamma * gae_lambda * not_done * running
            advantages[index] = running
        return advantages, advantages + values

    def finalize(self, last_reward_value: Tensor, last_cost_value: Tensor) -> RolloutBatch:
        if not self.records:
            raise ValueError("Cannot finalize an empty rollout")
        stacked = {
            key: torch.stack([record[key] for record in self.records]) for key in self.records[0]
        }
        reward_advantages, returns = self._gae(
            stacked["reward"],
            stacked["reward_value"],
            stacked["done"],
            last_reward_value,
            self.gamma,
            self.gae_lambda,
        )
        cost_advantages, cost_returns = self._gae(
            stacked["cost"],
            stacked["cost_value"],
            stacked["done"],
            last_cost_value,
            self.gamma,
            self.gae_lambda,
        )
        return RolloutBatch(
            maps=stacked["map"],
            states=stacked["state"],
            recurrent_states=stacked["recurrent_state"],
            episode_starts=stacked["episode_start"],
            actions=stacked["action"],
            action_masks=stacked["action_mask"],
            rewards=stacked["reward"],
            costs=stacked["cost"],
            reward_values=stacked["reward_value"],
            cost_values=stacked["cost_value"],
            log_probabilities=stacked["log_probability"],
            reward_advantages=reward_advantages,
            cost_advantages=cost_advantages,
            returns=returns,
            cost_returns=cost_returns,
        )
