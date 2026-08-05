from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass
class VectorCostRolloutBatch:
    maps: Tensor
    states: Tensor
    recurrent_states: Tensor
    episode_starts: Tensor
    actions: Tensor
    action_masks: Tensor
    rewards: Tensor
    vector_costs: Tensor
    reward_values: Tensor
    vector_cost_values: Tensor
    log_probabilities: Tensor
    reward_advantages: Tensor
    vector_cost_advantages: Tensor
    returns: Tensor
    vector_cost_returns: Tensor


class RecurrentVectorCostBuffer:
    """Recurrent rollout storage with independent GAE for every cost vector."""

    def __init__(
        self,
        *,
        cost_count: int = 5,
        reward_gamma: float = 0.99,
        reward_gae_lambda: float = 0.95,
        cost_gamma: float = 0.99,
        cost_gae_lambda: float = 0.95,
    ) -> None:
        self.cost_count = cost_count
        self.reward_gamma = reward_gamma
        self.reward_gae_lambda = reward_gae_lambda
        self.cost_gamma = cost_gamma
        self.cost_gae_lambda = cost_gae_lambda
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
            "vector_cost",
            "reward_value",
            "vector_cost_value",
            "log_probability",
            "done",
        }
        missing = required - record.keys()
        if missing:
            raise ValueError(f"Missing vector rollout fields: {sorted(missing)}")
        if record["vector_cost"].shape != (self.cost_count,):
            raise ValueError("Vector cost has the wrong shape")
        if record["vector_cost_value"].shape != (self.cost_count,):
            raise ValueError("Vector cost value has the wrong shape")
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
            while not_done.ndim < next_value.ndim:
                not_done = not_done.unsqueeze(-1)
            delta = signal[index] + gamma * next_value * not_done - values[index]
            running = delta + gamma * gae_lambda * not_done * running
            advantages[index] = running
        return advantages, advantages + values

    def finalize(
        self, last_reward_value: Tensor, last_vector_cost_value: Tensor
    ) -> VectorCostRolloutBatch:
        if not self.records:
            raise ValueError("Cannot finalize an empty vector rollout")
        stacked = {
            key: torch.stack([record[key] for record in self.records]) for key in self.records[0]
        }
        reward_advantages, returns = self._gae(
            stacked["reward"],
            stacked["reward_value"],
            stacked["done"],
            last_reward_value,
            self.reward_gamma,
            self.reward_gae_lambda,
        )
        vector_advantages, vector_returns = self._gae(
            stacked["vector_cost"],
            stacked["vector_cost_value"],
            stacked["done"],
            last_vector_cost_value,
            self.cost_gamma,
            self.cost_gae_lambda,
        )
        return VectorCostRolloutBatch(
            maps=stacked["map"],
            states=stacked["state"],
            recurrent_states=stacked["recurrent_state"],
            episode_starts=stacked["episode_start"],
            actions=stacked["action"],
            action_masks=stacked["action_mask"],
            rewards=stacked["reward"],
            vector_costs=stacked["vector_cost"],
            reward_values=stacked["reward_value"],
            vector_cost_values=stacked["vector_cost_value"],
            log_probabilities=stacked["log_probability"],
            reward_advantages=reward_advantages,
            vector_cost_advantages=vector_advantages,
            returns=returns,
            vector_cost_returns=vector_returns,
        )
