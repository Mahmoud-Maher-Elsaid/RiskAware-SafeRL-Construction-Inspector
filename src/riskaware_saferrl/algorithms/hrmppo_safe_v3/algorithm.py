from __future__ import annotations

import copy
from dataclasses import dataclass

import torch
from torch import Tensor, nn

from riskaware_saferrl.algorithms.hrmppo_safe_v3.pid import (
    AntiWindupPIDMultiplier,
)
from riskaware_saferrl.algorithms.hrmppo_safe_v3.policy import (
    RecurrentMaskedSafePolicyV3,
)
from riskaware_saferrl.buffers import VectorCostRolloutBatch
from riskaware_saferrl.safety.safety_contract_v3 import VECTOR_COST_NAMES


@dataclass(frozen=True)
class SafeV3Update:
    total_loss: float
    policy_loss: float
    reward_value_loss: float
    cost_value_losses: dict[str, float]
    cost_advantage_means: dict[str, float]
    multipliers: dict[str, float]
    multiplier_saturation: dict[str, bool]
    constraint_residuals: dict[str, float]
    hierarchy_loss: float
    entropy: float
    anchor_kl: float
    approximate_kl: float
    clip_fraction: float
    gradient_norm: float


class RiskShieldHRMPPOSafeV3:
    """Recurrent PPO with five independently constrained safety objectives."""

    def __init__(
        self,
        policy: RecurrentMaskedSafePolicyV3,
        *,
        budgets: dict[str, float],
        cost_scales: dict[str, float] | None = None,
        learning_rate: float = 5e-6,
        cost_critic_learning_rate: float = 1e-5,
        clip_range: float = 0.2,
        reward_value_coefficient: float = 0.1,
        cost_value_coefficient: float = 0.1,
        hierarchy_coefficient: float = 0.05,
        entropy_coefficient: float = 0.005,
        anchor_kl_coefficient: float = 1.0,
        maximum_kl: float = 0.03,
        max_gradient_norm: float = 0.5,
        multiplier_warmup_updates: int = 10,
    ) -> None:
        missing = set(VECTOR_COST_NAMES) - budgets.keys()
        if missing:
            raise ValueError(f"Missing vector safety budgets: {sorted(missing)}")
        self.policy = policy
        self.budgets = {name: float(budgets[name]) for name in VECTOR_COST_NAMES}
        self.cost_scales = {
            name: float((cost_scales or {}).get(name, max(self.budgets[name], 1.0)))
            for name in VECTOR_COST_NAMES
        }
        self.clip_range = clip_range
        self.reward_value_coefficient = reward_value_coefficient
        self.cost_value_coefficient = cost_value_coefficient
        self.hierarchy_coefficient = hierarchy_coefficient
        self.entropy_coefficient = entropy_coefficient
        self.anchor_kl_coefficient = anchor_kl_coefficient
        self.maximum_kl = maximum_kl
        self.max_gradient_norm = max_gradient_norm
        critic_parameters = list(policy.reward_critic.parameters())
        critic_parameters.extend(policy.cost_critics.parameters())
        critic_ids = {id(parameter) for parameter in critic_parameters}
        actor_parameters = [
            parameter for parameter in policy.parameters() if id(parameter) not in critic_ids
        ]
        self.optimizer = torch.optim.Adam(
            [
                {"params": actor_parameters, "lr": learning_rate},
                {"params": critic_parameters, "lr": cost_critic_learning_rate},
            ]
        )
        self.anchor_policy = copy.deepcopy(policy).eval()
        for parameter in self.anchor_policy.parameters():
            parameter.requires_grad_(False)
        self.policy.recurrent.flatten_parameters()
        self.anchor_policy.recurrent.flatten_parameters()
        self.multipliers = {
            name: AntiWindupPIDMultiplier(
                maximum=10.0 if name in {"collision", "restricted_zone"} else 5.0,
                warmup_updates=multiplier_warmup_updates,
                proportional_gain=0.08 if name in {"collision", "restricted_zone"} else 0.03,
                integral_gain=0.001,
                derivative_gain=0.01,
            )
            for name in VECTOR_COST_NAMES
        }

    @staticmethod
    def _normalize_per_cost(values: Tensor) -> Tensor:
        mean = values.mean(dim=(0,))
        std = values.std(dim=(0,), unbiased=False)
        return (values - mean) / (std + 1e-8)

    def update(
        self,
        batch: VectorCostRolloutBatch,
        *,
        observed_episode_costs: dict[str, float],
        epochs: int = 2,
    ) -> SafeV3Update:
        maps = batch.maps.unsqueeze(0)
        states = batch.states.unsqueeze(0)
        masks = batch.action_masks.unsqueeze(0).bool()
        actions = batch.actions.unsqueeze(0).long()
        episode_starts = batch.episode_starts.unsqueeze(0).bool()
        initial_hidden = batch.recurrent_states[0]
        old_log_probabilities = batch.log_probabilities.unsqueeze(0)
        reward_advantages = (batch.reward_advantages - batch.reward_advantages.mean()) / (
            batch.reward_advantages.std(unbiased=False) + 1e-8
        )
        reward_advantages = reward_advantages.unsqueeze(0)
        vector_advantages = self._normalize_per_cost(batch.vector_cost_advantages).unsqueeze(0)
        returns = batch.returns.unsqueeze(0)
        vector_returns = batch.vector_cost_returns.unsqueeze(0)
        subgoal_targets = torch.clamp(
            (states[..., 5] * self.policy.subgoal_count).long(),
            min=0,
            max=self.policy.subgoal_count - 1,
        )
        with torch.no_grad():
            anchor = self.anchor_policy(
                maps,
                states,
                masks,
                initial_hidden,
                episode_starts,
            )
        latest: dict[str, Tensor] = {}
        gradient_norm = torch.tensor(0.0, device=maps.device)
        for _ in range(epochs):
            output = self.policy(
                maps,
                states,
                masks,
                initial_hidden,
                episode_starts,
            )
            log_probabilities = output.distribution.log_prob(actions)
            ratio = torch.exp(log_probabilities - old_log_probabilities)
            clipped_ratio = torch.clamp(ratio, 1.0 - self.clip_range, 1.0 + self.clip_range)
            reward_surrogate = torch.minimum(
                ratio * reward_advantages,
                clipped_ratio * reward_advantages,
            )
            constrained = reward_surrogate
            for index, name in enumerate(VECTOR_COST_NAMES):
                cost_surrogate = torch.maximum(
                    ratio * vector_advantages[..., index],
                    clipped_ratio * vector_advantages[..., index],
                )
                constrained = constrained - self.multipliers[name].value * cost_surrogate
            policy_loss = -constrained.mean()
            reward_value_loss = nn.functional.mse_loss(output.reward_value, returns)
            cost_value_losses = torch.stack(
                [
                    nn.functional.mse_loss(
                        output.vector_cost_values[..., index],
                        vector_returns[..., index],
                    )
                    for index in range(len(VECTOR_COST_NAMES))
                ]
            )
            hierarchy_loss = nn.functional.cross_entropy(
                output.subgoal_logits.reshape(-1, self.policy.subgoal_count),
                subgoal_targets.reshape(-1),
            )
            entropy = output.distribution.entropy().mean()
            anchor_kl = torch.distributions.kl_divergence(
                anchor.distribution, output.distribution
            ).mean()
            loss = (
                policy_loss
                + self.reward_value_coefficient * reward_value_loss
                + self.cost_value_coefficient * cost_value_losses.sum()
                + self.hierarchy_coefficient * hierarchy_loss
                + self.anchor_kl_coefficient * anchor_kl
                - self.entropy_coefficient * entropy
            )
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            gradient_norm = nn.utils.clip_grad_norm_(
                self.policy.parameters(), self.max_gradient_norm
            )
            self.optimizer.step()
            with torch.no_grad():
                log_ratio = log_probabilities - old_log_probabilities
                approximate_kl = ((torch.exp(log_ratio) - 1.0) - log_ratio).mean()
                latest = {
                    "loss": loss,
                    "policy": policy_loss,
                    "reward_value": reward_value_loss,
                    "cost_values": cost_value_losses,
                    "hierarchy": hierarchy_loss,
                    "entropy": entropy,
                    "anchor_kl": anchor_kl,
                    "approximate_kl": approximate_kl,
                    "clip": ((ratio - 1.0).abs() > self.clip_range).float().mean(),
                }
            if float(approximate_kl.detach().cpu()) > self.maximum_kl:
                break
        residuals = {}
        for name in VECTOR_COST_NAMES:
            observed = float(observed_episode_costs[name])
            residuals[name] = observed - self.budgets[name]
            self.multipliers[name].update(
                observed,
                self.budgets[name],
                self.cost_scales[name],
            )
        cost_losses = latest["cost_values"]
        return SafeV3Update(
            total_loss=float(latest["loss"].detach().cpu()),
            policy_loss=float(latest["policy"].detach().cpu()),
            reward_value_loss=float(latest["reward_value"].detach().cpu()),
            cost_value_losses={
                name: float(cost_losses[index].detach().cpu())
                for index, name in enumerate(VECTOR_COST_NAMES)
            },
            cost_advantage_means={
                name: float(batch.vector_cost_advantages[..., index].mean().detach().cpu())
                for index, name in enumerate(VECTOR_COST_NAMES)
            },
            multipliers={name: multiplier.value for name, multiplier in self.multipliers.items()},
            multiplier_saturation={
                name: multiplier.saturated for name, multiplier in self.multipliers.items()
            },
            constraint_residuals=residuals,
            hierarchy_loss=float(latest["hierarchy"].detach().cpu()),
            entropy=float(latest["entropy"].detach().cpu()),
            anchor_kl=float(latest["anchor_kl"].detach().cpu()),
            approximate_kl=float(latest["approximate_kl"].detach().cpu()),
            clip_fraction=float(latest["clip"].detach().cpu()),
            gradient_norm=float(gradient_norm.detach().cpu()),
        )

    def state_dict(self) -> dict[str, object]:
        return {
            "algorithm": "RiskShield-HRMPPO-Safe-v3",
            "model_state_dict": self.policy.state_dict(),
            "anchor_policy_state_dict": self.anchor_policy.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "multipliers": {
                name: multiplier.state_dict() for name, multiplier in self.multipliers.items()
            },
            "budgets": self.budgets,
            "cost_scales": self.cost_scales,
        }

    def load_state_dict(self, state: dict[str, object]) -> None:
        self.policy.load_state_dict(state["model_state_dict"])  # type: ignore[arg-type]
        self.anchor_policy.load_state_dict(  # type: ignore[arg-type]
            state["anchor_policy_state_dict"]
        )
        self.optimizer.load_state_dict(state["optimizer_state_dict"])  # type: ignore[arg-type]
        multiplier_states = state["multipliers"]
        if not isinstance(multiplier_states, dict):
            raise TypeError("Multiplier checkpoint state must be a dictionary")
        for name, multiplier in self.multipliers.items():
            multiplier.load_state_dict(multiplier_states[name])
        self.policy.recurrent.flatten_parameters()
        self.anchor_policy.recurrent.flatten_parameters()
