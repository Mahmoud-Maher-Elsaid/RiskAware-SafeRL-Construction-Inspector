from __future__ import annotations

from typing import Any

import gymnasium as gym


class SafetyShield(gym.Wrapper):
    """Replace an unsafe proposed action with a currently safe action."""

    def step(
        self,
        action: int,
    ) -> tuple[dict[str, Any], float, bool, bool, dict[str, Any]]:
        original_action = int(action)
        executed_action = original_action
        shield_active = False

        base_env = self.unwrapped

        if hasattr(base_env, "is_action_safe") and not base_env.is_action_safe(original_action):
            safe_actions = base_env.safe_actions()
            executed_action = 4 if 4 in safe_actions else safe_actions[0]
            shield_active = True

        observation, reward, terminated, truncated, info = self.env.step(executed_action)
        info = dict(info)
        info["shield_active"] = shield_active
        info["proposed_action"] = original_action
        info["executed_action"] = executed_action

        return observation, reward, terminated, truncated, info