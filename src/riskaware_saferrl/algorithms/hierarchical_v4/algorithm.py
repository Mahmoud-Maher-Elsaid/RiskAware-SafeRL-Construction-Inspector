from __future__ import annotations

import copy
from dataclasses import dataclass

import torch
from torch import Tensor, nn

from riskaware_saferrl.algorithms.hrmppo_safe_v3.pid import AntiWindupPIDMultiplier
from riskaware_saferrl.buffers import VectorCostRolloutBatch
from riskaware_saferrl.hierarchical.policy import HierarchicalMissionPolicy
from riskaware_saferrl.hierarchical.schemas import VECTOR_COST_NAMES


@dataclass(frozen=True)
class HierarchicalV4Update:
    total_loss: float
    option_policy_loss: float
    reward_value_loss: float
    cost_value_losses: dict[str, float]
    multipliers: dict[str, float]
    constraint_residuals: dict[str, float]
    entropy: float
    anchor_kl: float
    approximate_kl: float
    clip_fraction: float
    gradient_norm: float


class RiskShieldHierarchicalHRMPPOV4:
    """Recurrent constrained PPO over mission options, not motor primitives."""

    def __init__(
        self,
        policy: HierarchicalMissionPolicy,
        *,
        budgets: dict[str, float],
        cost_scales: dict[str, float] | None = None,
        learning_rate: float = 3e-6,
        cost_critic_learning_rate: float = 1e-5,
        clip_range: float = 0.15,
        reward_value_coefficient: float = 0.25,
        cost_value_coefficient: float = 0.2,
        entropy_coefficient: float = 0.002,
        anchor_kl_coefficient: float = 1.0,
        maximum_kl: float = 0.02,
        maximum_gradient_norm: float = 0.5,
        multiplier_warmup_updates: int = 10,
    ) -> None:
        missing = set(VECTOR_COST_NAMES) - budgets.keys()
        if missing:
            raise ValueError(f"Missing hierarchical safety budgets: {sorted(missing)}")
        self.policy = policy
        self.anchor_policy = copy.deepcopy(policy).eval()
        for parameter in self.anchor_policy.parameters():
            parameter.requires_grad_(False)
        self.budgets = {name: float(budgets[name]) for name in VECTOR_COST_NAMES}
        self.cost_scales = {
            name: float((cost_scales or {}).get(name, max(self.budgets[name], 1.0)))
            for name in VECTOR_COST_NAMES
        }
        self.clip_range = clip_range
        self.reward_value_coefficient = reward_value_coefficient
        self.cost_value_coefficient = cost_value_coefficient
        self.entropy_coefficient = entropy_coefficient
        self.anchor_kl_coefficient = anchor_kl_coefficient
        self.maximum_kl = maximum_kl
        self.maximum_gradient_norm = maximum_gradient_norm
        critic_parameters = list(policy.reward_critic.parameters())
        critic_parameters.extend(policy.cost_critics.parameters())
        critic_ids = {id(parameter) for parameter in critic_parameters}
        actor_parameters = [
            parameter for parameter in policy.parameters() if id(parameter) not in critic_ids
        ]
        self.optimizer = torch.optim.AdamW(
            [
                {"params": actor_parameters, "lr": learning_rate},
                {"params": critic_parameters, "lr": cost_critic_learning_rate},
            ],
            weight_decay=1e-5,
        )
        hard = {"collision", "restricted_zone", "human_clearance"}
        self.multipliers = {
            name: AntiWindupPIDMultiplier(
                maximum=10.0 if name in hard else 4.0,
                warmup_updates=multiplier_warmup_updates,
                proportional_gain=0.08 if name in hard else 0.02,
                integral_gain=0.001,
                derivative_gain=0.01,
            )
            for name in VECTOR_COST_NAMES
        }

    @staticmethod
    def _normalize(values: Tensor) -> Tensor:
        return (values - values.mean(dim=0)) / (values.std(dim=0, unbiased=False) + 1e-8)

    def update(
        self,
        batch: VectorCostRolloutBatch,
        *,
        observed_episode_costs: dict[str, float],
        epochs: int = 2,
    ) -> HierarchicalV4Update:
        maps = batch.maps.unsqueeze(0)
        states = batch.states.unsqueeze(0)
        masks = batch.action_masks.unsqueeze(0).bool()
        options = batch.actions.unsqueeze(0).long()
        episode_starts = batch.episode_starts.unsqueeze(0).bool()
        initial_hidden = batch.recurrent_states[0]
        old_log_probabilities = batch.log_probabilities.unsqueeze(0)
        reward_advantages = self._normalize(batch.reward_advantages).unsqueeze(0)
        cost_advantages = self._normalize(batch.vector_cost_advantages).unsqueeze(0)
        reward_returns = batch.returns.unsqueeze(0)
        cost_returns = batch.vector_cost_returns.unsqueeze(0)
        with torch.no_grad():
            anchor = self.anchor_policy(maps, states, masks, initial_hidden, episode_starts)
        latest: dict[str, Tensor] = {}
        gradient_norm = torch.zeros((), device=maps.device)
        for _ in range(epochs):
            output = self.policy(maps, states, masks, initial_hidden, episode_starts)
            log_probabilities = output.option_distribution.log_prob(options)
            ratio = torch.exp(log_probabilities - old_log_probabilities)
            clipped = torch.clamp(ratio, 1.0 - self.clip_range, 1.0 + self.clip_range)
            objective = torch.minimum(
                ratio * reward_advantages,
                clipped * reward_advantages,
            )
            for index, name in enumerate(VECTOR_COST_NAMES):
                cost_objective = torch.maximum(
                    ratio * cost_advantages[..., index],
                    clipped * cost_advantages[..., index],
                )
                objective = objective - self.multipliers[name].value * cost_objective
            policy_loss = -objective.mean()
            reward_value_loss = nn.functional.smooth_l1_loss(output.reward_value, reward_returns)
            cost_value_losses = torch.stack(
                [
                    nn.functional.smooth_l1_loss(
                        output.vector_cost_values[..., index],
                        cost_returns[..., index],
                    )
                    for index in range(len(VECTOR_COST_NAMES))
                ]
            )
            entropy = output.option_distribution.entropy().mean()
            anchor_kl = torch.distributions.kl_divergence(
                anchor.option_distribution,
                output.option_distribution,
            ).mean()
            loss = (
                policy_loss
                + self.reward_value_coefficient * reward_value_loss
                + self.cost_value_coefficient * cost_value_losses.sum()
                + self.anchor_kl_coefficient * anchor_kl
                - self.entropy_coefficient * entropy
            )
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            gradient_norm = nn.utils.clip_grad_norm_(
                self.policy.parameters(), self.maximum_gradient_norm
            )
            self.optimizer.step()
            with torch.no_grad():
                log_ratio = log_probabilities - old_log_probabilities
                approximate_kl = ((torch.exp(log_ratio) - 1.0) - log_ratio).mean()
                latest = {
                    "loss": loss,
                    "policy_loss": policy_loss,
                    "reward_value_loss": reward_value_loss,
                    "cost_value_losses": cost_value_losses,
                    "entropy": entropy,
                    "anchor_kl": anchor_kl,
                    "approximate_kl": approximate_kl,
                    "clip_fraction": ((ratio - 1.0).abs() > self.clip_range).float().mean(),
                }
            if float(approximate_kl) > self.maximum_kl:
                break
        residuals: dict[str, float] = {}
        for name in VECTOR_COST_NAMES:
            observed = float(observed_episode_costs[name])
            residuals[name] = observed - self.budgets[name]
            self.multipliers[name].update(observed, self.budgets[name], self.cost_scales[name])
        losses = latest["cost_value_losses"]
        return HierarchicalV4Update(
            total_loss=float(latest["loss"].detach().cpu()),
            option_policy_loss=float(latest["policy_loss"].detach().cpu()),
            reward_value_loss=float(latest["reward_value_loss"].detach().cpu()),
            cost_value_losses={
                name: float(losses[index].detach().cpu())
                for index, name in enumerate(VECTOR_COST_NAMES)
            },
            multipliers={name: multiplier.value for name, multiplier in self.multipliers.items()},
            constraint_residuals=residuals,
            entropy=float(latest["entropy"].detach().cpu()),
            anchor_kl=float(latest["anchor_kl"].detach().cpu()),
            approximate_kl=float(latest["approximate_kl"].detach().cpu()),
            clip_fraction=float(latest["clip_fraction"].detach().cpu()),
            gradient_norm=float(gradient_norm.detach().cpu()),
        )

    def state_dict(self) -> dict[str, object]:
        return {
            "algorithm": "RiskShield-Hierarchical-HRMPPO-MPC-v4",
            "model": self.policy.state_dict(),
            "anchor_model": self.anchor_policy.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "multipliers": {
                name: multiplier.state_dict() for name, multiplier in self.multipliers.items()
            },
            "budgets": self.budgets,
            "cost_scales": self.cost_scales,
        }

    def load_state_dict(self, state: dict[str, object]) -> None:
        self.policy.load_state_dict(state["model"])  # type: ignore[arg-type]
        self.anchor_policy.load_state_dict(state["anchor_model"])  # type: ignore[arg-type]
        self.optimizer.load_state_dict(state["optimizer"])  # type: ignore[arg-type]
        multiplier_states = state["multipliers"]
        if not isinstance(multiplier_states, dict):
            raise TypeError("Multiplier state must be a dictionary")
        for name, multiplier in self.multipliers.items():
            multiplier.load_state_dict(multiplier_states[name])
