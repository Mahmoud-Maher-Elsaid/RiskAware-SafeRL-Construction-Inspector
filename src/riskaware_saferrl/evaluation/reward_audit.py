from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np

REWARD_COMPONENT_NAMES = (
    "step_penalty",
    "new_cell_reward",
    "hazard_reward",
    "completion_reward",
    "collision_penalty",
    "worker_penalty",
    "restricted_penalty",
    "invalid_inspection_penalty",
)


class RewardAuditWrapper(gym.Wrapper):
    """Reconstruct and expose the environment reward components."""

    def step(
        self,
        action: int,
    ) -> tuple[dict[str, np.ndarray], float, bool, bool, dict[str, Any]]:
        previous_coverage = len(self.unwrapped.visited)

        observation, reward, terminated, truncated, info = self.env.step(action)

        components = {
            "step_penalty": -0.01,
            "new_cell_reward": (0.05 if len(self.unwrapped.visited) > previous_coverage else 0.0),
            "hazard_reward": (3.0 if bool(info.get("new_hazard", False)) else 0.0),
            "completion_reward": 5.0 if terminated else 0.0,
            "collision_penalty": (-1.0 if float(info.get("cost_collision", 0.0)) > 0.0 else 0.0),
            "worker_penalty": (-0.5 if float(info.get("cost_worker", 0.0)) > 0.0 else 0.0),
            "restricted_penalty": (-0.75 if float(info.get("cost_restricted", 0.0)) > 0.0 else 0.0),
            "invalid_inspection_penalty": (
                -0.1 if action == 4 and not bool(info.get("new_hazard", False)) else 0.0
            ),
        }

        reconstructed_reward = sum(components.values())

        if not np.isclose(
            reconstructed_reward,
            reward,
            atol=1e-8,
        ):
            raise RuntimeError(
                "Reward reconstruction mismatch: "
                f"environment={reward}, "
                f"reconstructed={reconstructed_reward}"
            )

        info = dict(info)
        info["reward_components"] = components
        return observation, reward, terminated, truncated, info
