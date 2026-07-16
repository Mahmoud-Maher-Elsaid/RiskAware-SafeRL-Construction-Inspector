from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np

from riskaware_saferrl.safety.lagrangian import (
    LagrangeMultiplier,
)


class CounterfactualLagrangianReward(gym.Wrapper):
    """Penalize unsafe policy proposals before shield replacement."""

    COST_KEYS = {
        "collision": "proposed_cost_collision",
        "worker": "proposed_cost_worker",
        "restricted": "proposed_cost_restricted",
        "invalid_action": "proposed_cost_invalid_action",
        "unsafe": "proposed_cost_unsafe",
    }

    def __init__(
        self,
        environment: gym.Env,
        multiplier: LagrangeMultiplier,
    ) -> None:
        super().__init__(environment)
        self.multiplier = multiplier
        self.episode_proposed_cost = 0.0

    def action_masks(self) -> np.ndarray:
        mask_provider = getattr(
            self.env,
            "action_masks",
            None,
        )

        if mask_provider is None or not callable(mask_provider):
            raise AttributeError("The wrapped environment does not provide action_masks().")

        return np.asarray(
            mask_provider(),
            dtype=np.bool_,
        ).copy()

    def _violations(
        self,
        action: int,
    ) -> tuple[str, ...]:
        provider = getattr(
            self.unwrapped,
            "action_safety_violations",
            None,
        )

        if provider is None or not callable(provider):
            raise AttributeError(
                "The base environment does not provide action_safety_violations()."
            )

        return tuple(str(value) for value in provider(action))

    def reset(
        self,
        **kwargs: Any,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        self.episode_proposed_cost = 0.0
        return self.env.reset(**kwargs)

    def step(
        self,
        action: int,
    ) -> tuple[
        dict[str, Any],
        float,
        bool,
        bool,
        dict[str, Any],
    ]:
        proposed_action = int(action)
        violations = self._violations(proposed_action)
        cost_components = {key: 0.0 for key in self.COST_KEYS.values()}

        for violation in violations:
            key = self.COST_KEYS.get(
                violation,
                "proposed_cost_unsafe",
            )
            cost_components[key] += 1.0

        proposed_cost = float(sum(cost_components.values()))
        multiplier_value = float(self.multiplier.value)

        (
            observation,
            task_reward,
            terminated,
            truncated,
            info,
        ) = self.env.step(proposed_action)

        constrained_reward = self.multiplier.penalized_reward(
            float(task_reward),
            proposed_cost,
        )
        self.episode_proposed_cost += proposed_cost

        enriched_info = dict(info)
        enriched_info.update(cost_components)
        enriched_info["task_reward"] = float(task_reward)
        enriched_info["proposed_action_cost"] = proposed_cost
        enriched_info["lagrange_multiplier"] = multiplier_value
        enriched_info["lagrangian_penalty"] = multiplier_value * proposed_cost
        enriched_info["constrained_reward"] = float(constrained_reward)

        if terminated or truncated:
            enriched_info["episode_proposed_action_cost"] = float(self.episode_proposed_cost)

        return (
            observation,
            float(constrained_reward),
            terminated,
            truncated,
            enriched_info,
        )
