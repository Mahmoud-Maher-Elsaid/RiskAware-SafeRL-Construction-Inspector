from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np


class SafetyShield(gym.Wrapper):
    "Project unsafe actions onto safe task-valid actions."

    def action_masks(self) -> np.ndarray:
        mask_provider = getattr(self.unwrapped, "action_masks", None)

        if mask_provider is None or not callable(mask_provider):
            raise AttributeError("The wrapped environment does not provide action_masks().")

        return np.asarray(
            mask_provider(),
            dtype=np.bool_,
        ).copy()

    def _safety_violations(self, action: int) -> tuple[str, ...]:
        base_environment = self.unwrapped
        violation_provider = getattr(
            base_environment,
            "action_safety_violations",
            None,
        )

        if violation_provider is not None and callable(violation_provider):
            return tuple(str(value) for value in violation_provider(action))

        safety_provider = getattr(
            base_environment,
            "is_action_safe",
            None,
        )

        if safety_provider is None or not callable(safety_provider):
            return ()

        return () if bool(safety_provider(action)) else ("unsafe",)

    def _task_valid_actions(self) -> list[int]:
        base_environment = self.unwrapped
        action_provider = getattr(
            base_environment,
            "task_valid_actions",
            None,
        )

        if action_provider is not None and callable(action_provider):
            return [int(action) for action in action_provider()]

        return list(range(base_environment.action_space.n))

    def _safe_task_valid_actions(self) -> list[int]:
        base_environment = self.unwrapped

        return [
            action
            for action in self._task_valid_actions()
            if bool(base_environment.is_action_safe(action))
        ]

    def _projection_priority(
        self,
        proposed_action: int,
        candidate_action: int,
    ) -> tuple[float, ...]:
        base_environment = self.unwrapped
        proposed_delta = base_environment.ACTION_TO_DELTA.get(proposed_action)
        candidate_delta = base_environment.ACTION_TO_DELTA.get(candidate_action)

        if candidate_delta is None:
            return (
                1.0,
                0.0,
                float(candidate_action),
            )

        if proposed_delta is None:
            return (
                0.0,
                0.0,
                float(candidate_action),
            )

        deviation = float(np.sum(np.square(proposed_delta - candidate_delta)))

        return (
            0.0,
            deviation,
            float(candidate_action),
        )

    def replacement_action(
        self,
        proposed_action: int,
    ) -> int:
        candidates = self._safe_task_valid_actions()

        if not candidates:
            raise RuntimeError("The semantic safety shield found no safe task-valid action.")

        return min(
            candidates,
            key=lambda candidate_action: self._projection_priority(
                proposed_action,
                candidate_action,
            ),
        )

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
        violations = self._safety_violations(proposed_action)
        shield_active = bool(violations)
        executed_action = (
            self.replacement_action(proposed_action) if shield_active else proposed_action
        )
        replacement_safe = bool(self.unwrapped.is_action_safe(executed_action))
        replacement_task_valid = executed_action in self._task_valid_actions()

        observation, reward, terminated, truncated, info = self.env.step(executed_action)

        enriched_info = dict(info)
        enriched_info["shield_active"] = shield_active
        enriched_info["shield_violations"] = list(violations)
        enriched_info["shield_reason"] = violations[0] if violations else None
        enriched_info["proposed_action"] = proposed_action
        enriched_info["executed_action"] = executed_action
        enriched_info["shield_replacement_safe"] = replacement_safe
        enriched_info["shield_replacement_task_valid"] = replacement_task_valid

        return (
            observation,
            reward,
            terminated,
            truncated,
            enriched_info,
        )
