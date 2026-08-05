from __future__ import annotations

import copy
from dataclasses import dataclass

import torch
from torch import Tensor, nn

from riskaware_saferrl.buffers import RolloutBatch
from riskaware_saferrl.policies import RecurrentMaskedPolicy


@dataclass
class PIDLagrangeController:
    """Projected PID controller for a non-negative constraint multiplier."""

    value: float = 0.0
    proportional_gain: float = 0.05
    integral_gain: float = 0.002
    derivative_gain: float = 0.01
    maximum: float = 25.0
    integral: float = 0.0
    previous_error: float = 0.0

    def update(self, observed_cost: float, safety_budget: float) -> float:
        error = observed_cost - safety_budget
        self.integral = max(-1000.0, min(1000.0, self.integral + error))
        derivative = error - self.previous_error
        self.previous_error = error
        update = (
            self.proportional_gain * error
            + self.integral_gain * self.integral
            + self.derivative_gain * derivative
        )
        self.value = max(0.0, min(self.maximum, self.value + update))
        return self.value

    def state_dict(self) -> dict[str, float]:
        return {
            "value": self.value,
            "integral": self.integral,
            "previous_error": self.previous_error,
        }

    def load_state_dict(self, state: dict[str, float]) -> None:
        self.value = float(state["value"])
        self.integral = float(state["integral"])
        self.previous_error = float(state["previous_error"])


@dataclass(frozen=True)
class HRMPPOUpdate:
    total_loss: float
    actor_loss: float
    reward_value_loss: float
    cost_value_loss: float
    hierarchy_loss: float
    anchor_kl: float
    entropy: float
    approximate_kl: float
    clip_fraction: float
    lagrange_multiplier: float


class RiskShieldHRMPPOV2:
    """Recurrent masked PPO with separate cost GAE and a PID constraint."""

    def __init__(
        self,
        policy: RecurrentMaskedPolicy,
        *,
        learning_rate: float = 3e-4,
        clip_range: float = 0.2,
        value_coefficient: float = 0.5,
        cost_value_coefficient: float = 0.5,
        hierarchy_coefficient: float = 0.05,
        anchor_kl_coefficient: float = 1.0,
        entropy_coefficient: float = 0.01,
        max_gradient_norm: float = 0.5,
        safety_budget: float = 20.0,
        pid_lagrange: PIDLagrangeController | None = None,
    ) -> None:
        self.policy = policy
        self.clip_range = clip_range
        self.value_coefficient = value_coefficient
        self.cost_value_coefficient = cost_value_coefficient
        self.hierarchy_coefficient = hierarchy_coefficient
        self.anchor_kl_coefficient = anchor_kl_coefficient
        self.entropy_coefficient = entropy_coefficient
        self.max_gradient_norm = max_gradient_norm
        self.safety_budget = safety_budget
        self.pid_lagrange = pid_lagrange or PIDLagrangeController()
        self.optimizer = torch.optim.Adam(policy.parameters(), lr=learning_rate)
        self.anchor_policy = copy.deepcopy(policy).eval()
        for parameter in self.anchor_policy.parameters():
            parameter.requires_grad_(False)
        self.policy.recurrent.flatten_parameters()
        self.anchor_policy.recurrent.flatten_parameters()

    @staticmethod
    def _normalize(advantages: Tensor) -> Tensor:
        return (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)

    def update(
        self,
        batch: RolloutBatch,
        *,
        epochs: int = 4,
        observed_episode_cost: float | None = None,
    ) -> HRMPPOUpdate:
        maps = batch.maps.unsqueeze(0)
        states = batch.states.unsqueeze(0)
        masks = batch.action_masks.unsqueeze(0).bool()
        actions = batch.actions.unsqueeze(0).long()
        episode_starts = batch.episode_starts.unsqueeze(0).bool()
        initial_hidden = batch.recurrent_states[0]
        old_log_probabilities = batch.log_probabilities.unsqueeze(0)
        reward_advantages = self._normalize(batch.reward_advantages).unsqueeze(0)
        cost_advantages = self._normalize(batch.cost_advantages).unsqueeze(0)
        returns = batch.returns.unsqueeze(0)
        cost_returns = batch.cost_returns.unsqueeze(0)
        # Progress-derived auxiliary labels are causal state features, not oracle goals.
        subgoal_targets = torch.clamp(
            (states[..., 5] * self.policy.subgoal_count).long(),
            min=0,
            max=self.policy.subgoal_count - 1,
        )
        latest: dict[str, Tensor] = {}
        with torch.no_grad():
            anchor_output = self.anchor_policy(
                maps,
                states,
                masks,
                initial_hidden,
                episode_starts=episode_starts,
            )
        for _ in range(epochs):
            output = self.policy(
                maps,
                states,
                masks,
                initial_hidden,
                episode_starts=episode_starts,
            )
            log_probabilities = output.distribution.log_prob(actions)
            ratio = torch.exp(log_probabilities - old_log_probabilities)
            reward_surrogate = torch.minimum(
                ratio * reward_advantages,
                torch.clamp(ratio, 1.0 - self.clip_range, 1.0 + self.clip_range)
                * reward_advantages,
            )
            cost_surrogate = torch.maximum(
                ratio * cost_advantages,
                torch.clamp(ratio, 1.0 - self.clip_range, 1.0 + self.clip_range) * cost_advantages,
            )
            actor_loss = -(reward_surrogate - self.pid_lagrange.value * cost_surrogate).mean()
            reward_value_loss = nn.functional.mse_loss(output.reward_value, returns)
            cost_value_loss = nn.functional.mse_loss(output.cost_value, cost_returns)
            hierarchy_loss = nn.functional.cross_entropy(
                output.subgoal_logits.reshape(-1, self.policy.subgoal_count),
                subgoal_targets.reshape(-1),
            )
            entropy = output.distribution.entropy().mean()
            anchor_kl = torch.distributions.kl_divergence(
                anchor_output.distribution, output.distribution
            ).mean()
            loss = (
                actor_loss
                + self.value_coefficient * reward_value_loss
                + self.cost_value_coefficient * cost_value_loss
                + self.hierarchy_coefficient * hierarchy_loss
                + self.anchor_kl_coefficient * anchor_kl
                - self.entropy_coefficient * entropy
            )
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_gradient_norm)
            self.optimizer.step()
            with torch.no_grad():
                log_ratio = log_probabilities - old_log_probabilities
                latest = {
                    "loss": loss,
                    "actor": actor_loss,
                    "reward_value": reward_value_loss,
                    "cost_value": cost_value_loss,
                    "hierarchy": hierarchy_loss,
                    "anchor_kl": anchor_kl,
                    "entropy": entropy,
                    "kl": ((torch.exp(log_ratio) - 1.0) - log_ratio).mean(),
                    "clip": ((ratio - 1.0).abs() > self.clip_range).float().mean(),
                }
        observed_cost = (
            float(batch.costs.sum().detach().cpu())
            if observed_episode_cost is None
            else observed_episode_cost
        )
        self.pid_lagrange.update(observed_cost, self.safety_budget)
        return HRMPPOUpdate(
            total_loss=float(latest["loss"].detach().cpu()),
            actor_loss=float(latest["actor"].detach().cpu()),
            reward_value_loss=float(latest["reward_value"].detach().cpu()),
            cost_value_loss=float(latest["cost_value"].detach().cpu()),
            hierarchy_loss=float(latest["hierarchy"].detach().cpu()),
            anchor_kl=float(latest["anchor_kl"].detach().cpu()),
            entropy=float(latest["entropy"].detach().cpu()),
            approximate_kl=float(latest["kl"].detach().cpu()),
            clip_fraction=float(latest["clip"].detach().cpu()),
            lagrange_multiplier=self.pid_lagrange.value,
        )

    def state_dict(self) -> dict[str, object]:
        return {
            "model_state_dict": self.policy.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "pid_lagrange_state": self.pid_lagrange.state_dict(),
            "anchor_policy_state_dict": self.anchor_policy.state_dict(),
            "safety_budget": self.safety_budget,
            "algorithm": "RiskShield-HRMPPO-v2",
        }

    def load_state_dict(self, state: dict[str, object]) -> None:
        self.policy.load_state_dict(state["model_state_dict"])  # type: ignore[arg-type]
        self.optimizer.load_state_dict(state["optimizer_state_dict"])  # type: ignore[arg-type]
        self.pid_lagrange.load_state_dict(state["pid_lagrange_state"])  # type: ignore[arg-type]
        self.anchor_policy.load_state_dict(state["anchor_policy_state_dict"])  # type: ignore[arg-type]
        self.policy.recurrent.flatten_parameters()
        self.anchor_policy.recurrent.flatten_parameters()
