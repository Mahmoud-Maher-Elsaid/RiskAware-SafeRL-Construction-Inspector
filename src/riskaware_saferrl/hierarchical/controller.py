from __future__ import annotations

from dataclasses import dataclass

import numpy as np

Position = tuple[int, int]
ACTION_DELTAS = {0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1), 4: (0, 0)}


@dataclass(frozen=True)
class LocalControllerDecision:
    primitive: int
    predicted_trajectory: tuple[Position, ...]
    objective: float
    human_clearance_margin: float
    reason: str


class PredictiveLocalController:
    """Discrete short-horizon controller tracking a causal planner path."""

    def __init__(self, *, horizon: int = 3, continuity_weight: float = 0.1) -> None:
        if not 1 <= horizon <= 5:
            raise ValueError("Controller horizon must be between one and five")
        self.horizon = horizon
        self.continuity_weight = continuity_weight
        self._previous_primitive: int | None = None

    def reset(self) -> None:
        self._previous_primitive = None

    @staticmethod
    def _position(semantic_map: np.ndarray) -> Position:
        locations = np.argwhere(semantic_map[5] > 0)
        if len(locations) != 1:
            raise ValueError("Observation must contain exactly one robot cell")
        return int(locations[0, 0]), int(locations[0, 1])

    def decide(
        self,
        observation: dict[str, np.ndarray],
        path: tuple[Position, ...],
        *,
        inspection_intent: bool,
    ) -> LocalControllerDecision:
        semantic_map = np.asarray(observation["map"])
        mask = np.asarray(observation["action_mask"], dtype=np.bool_)
        position = self._position(semantic_map)
        desired = path[min(1, len(path) - 1)] if path else position
        workers = np.argwhere((semantic_map[2] > 0) | (semantic_map[7] > 0))
        candidates: list[tuple[float, int, tuple[Position, ...], float]] = []
        for primitive, delta in ACTION_DELTAS.items():
            if not mask[primitive]:
                continue
            candidate = position[0] + delta[0], position[1] + delta[1]
            trajectory = (position, candidate)
            clearance = min(
                (
                    abs(candidate[0] - int(row)) + abs(candidate[1] - int(column))
                    for row, column in workers
                ),
                default=99,
            )
            collision = (
                not 0 <= candidate[0] < semantic_map.shape[-1]
                or not 0 <= candidate[1] < semantic_map.shape[-1]
                or semantic_map[0, candidate[0], candidate[1]] > 0
            )
            restricted = not collision and semantic_map[3, candidate[0], candidate[1]] > 0
            hard = 1_000.0 * float(collision or restricted or clearance < 1)
            progress = abs(candidate[0] - desired[0]) + abs(candidate[1] - desired[1])
            risk = (
                0.0
                if collision
                else float(
                    max(
                        semantic_map[6, candidate[0], candidate[1]],
                        semantic_map[8, candidate[0], candidate[1]],
                    )
                )
            )
            if inspection_intent and primitive == 4:
                progress -= 1.0
            continuity = (
                self.continuity_weight
                if self._previous_primitive is not None and primitive != self._previous_primitive
                else 0.0
            )
            candidates.append(
                (
                    hard + float(progress) + 4.0 * risk + continuity,
                    primitive,
                    trajectory,
                    float(clearance),
                )
            )
        if not candidates:
            raise ValueError("Action mask contains no valid primitive")
        objective, primitive, trajectory, clearance = min(candidates)
        self._previous_primitive = primitive
        return LocalControllerDecision(
            primitive=primitive,
            predicted_trajectory=trajectory,
            objective=objective,
            human_clearance_margin=clearance,
            reason="minimum_predicted_vector_cost_and_path_error",
        )
