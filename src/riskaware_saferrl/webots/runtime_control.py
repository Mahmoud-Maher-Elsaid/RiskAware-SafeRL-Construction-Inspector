from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from riskaware_saferrl.webots.bridge import GridAction, GridPosition


@dataclass
class RuntimeSafetyEvaluator:
    """Stateful safety projection used by the dedicated Webots policy runtime."""

    forbidden_actions: frozenset[int] = frozenset()
    emergency_stop: bool = False
    intervention_count: int = 0

    def configure(
        self,
        *,
        forbidden_actions: set[int] | frozenset[int],
        emergency_stop: bool,
    ) -> None:
        self.forbidden_actions = frozenset(int(action) for action in forbidden_actions)
        self.emergency_stop = bool(emergency_stop)

    def evaluate(self, proposed_action: int) -> tuple[int, str, float]:
        action = int(proposed_action)
        if self.emergency_stop:
            self.intervention_count += 1
            return int(GridAction.INSPECT), "emergency_hold", 1.0
        if action in self.forbidden_actions:
            self.intervention_count += 1
            return int(GridAction.INSPECT), "safe_projection", 0.75
        return action, "not_needed", 0.0


def inject_perception_risk(
    observation: dict[str, np.ndarray],
    *,
    agent_position: GridPosition,
    grid_size: int,
    risk_score: float,
) -> dict[str, np.ndarray]:
    """Add live CV risk to the checkpoint-compatible semantic risk channel."""

    resolved = {
        "map": np.asarray(observation["map"], dtype=np.float32).copy(),
        "state": np.asarray(observation["state"], dtype=np.float32).copy(),
    }
    semantic_map = resolved["map"].reshape(7, grid_size, grid_size)
    row, column = agent_position
    semantic_map[6, row, column] = max(
        float(semantic_map[6, row, column]),
        float(np.clip(risk_score, 0.0, 1.0)),
    )
    return resolved
