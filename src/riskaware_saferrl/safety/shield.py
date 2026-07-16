from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np


class SafetyShield(gym.Wrapper):
    "Project unsafe actions onto the safest available action."

    EMERGENCY_HOLD_ACTION = 4

    VIOLATION_WEIGHTS = {
        "collision": 1000.0,
        "invalid_action": 1000.0,
        "restricted": 1.0,
        "worker": 1.0,
        "unsafe": 1.0,
    }

    def action_masks(self) -> np.ndarray:
        mask_provider = getattr(
            self.unwrapped,
            "action_masks",
            None,
        )

        if mask_provider is None or not callable(mask_provider):
            raise AttributeError("The wrapped environment does not provide action_masks().")

        return np.asarray(
            mask_provider(),
            dtype=np.bool_,
        ).copy()

    def _safety_violations(
        self,
        action: int,
    ) -> tuple[str, ...]:
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
        return [
            action for action in self._task_valid_actions() if not self._safety_violations(action)
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

    def _violation_score(
        self,
        action: int,
    ) -> float:
        return float(
            sum(
                self.VIOLATION_WEIGHTS.get(
                    violation,
                    1.0,
                )
                for violation in self._safety_violations(action)
            )
        )

    def replacement_decision(
        self,
        proposed_action: int,
    ) -> tuple[int, str]:
        safe_task_valid_actions = self._safe_task_valid_actions()

        if safe_task_valid_actions:
            action = min(
                safe_task_valid_actions,
                key=lambda candidate_action: self._projection_priority(
                    proposed_action,
                    candidate_action,
                ),
            )
            return action, "safe_projection"

        emergency_action = self.EMERGENCY_HOLD_ACTION

        if not self._safety_violations(emergency_action):
            return emergency_action, "emergency_hold"

        fallback_actions = self._task_valid_actions()

        if not fallback_actions:
            fallback_actions = [emergency_action]

        action = min(
            fallback_actions,
            key=lambda candidate_action: (
                self._violation_score(candidate_action),
                self._projection_priority(
                    proposed_action,
                    candidate_action,
                ),
            ),
        )
        return action, "least_unsafe"

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

        if shield_active:
            executed_action, resolution = self.replacement_decision(proposed_action)
        else:
            executed_action = proposed_action
            resolution = "not_needed"

        replacement_violations = self._safety_violations(executed_action)
        replacement_task_valid = executed_action in self._task_valid_actions()

        observation, reward, terminated, truncated, info = self.env.step(executed_action)

        enriched_info = dict(info)
        enriched_info["shield_active"] = shield_active
        enriched_info["shield_violations"] = list(violations)
        enriched_info["shield_reason"] = violations[0] if violations else None
        enriched_info["proposed_action"] = proposed_action
        enriched_info["executed_action"] = executed_action
        enriched_info["shield_resolution"] = resolution
        enriched_info["shield_emergency_hold"] = resolution == "emergency_hold"
        enriched_info["shield_unavoidable_violation"] = resolution == "least_unsafe"
        enriched_info["shield_replacement_violations"] = list(replacement_violations)
        enriched_info["shield_replacement_safe"] = not replacement_violations
        enriched_info["shield_replacement_task_valid"] = replacement_task_valid

        return (
            observation,
            reward,
            terminated,
            truncated,
            enriched_info,
        )
