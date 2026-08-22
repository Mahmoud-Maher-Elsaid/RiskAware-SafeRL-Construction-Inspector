from __future__ import annotations

from dataclasses import dataclass

import numpy as np

Position = tuple[int, int]
# The observation grid is robot-relative: row -1 is forward, column -1 is
# the robot's physical left, and column +1 is its physical right.  These are
# deliberately separate from the canonical MotionPrimitive ids (STOP=0,
# MOVE_FORWARD=1, TURN_LEFT=2, TURN_RIGHT=3, INSPECT=4).
GRID_MOTION = (
    (1, (-1, 0)),  # MOVE_FORWARD
    (2, (0, -1)),  # TURN_LEFT
    (3, (0, 1)),  # TURN_RIGHT
    (0, (0, 0)),  # STOP when the path is exhausted
)


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
        # A hold/emergency planner result is represented by an exhausted
        # one-cell path.  It must be an actual STOP; scoring lateral motions
        # here can turn a safety hold into an in-place spin.
        if len(path) <= 1 and not inspection_intent:
            self._previous_primitive = 0
            return LocalControllerDecision(
                primitive=0,
                predicted_trajectory=(position, position),
                objective=0.0,
                human_clearance_margin=99.0,
                reason="exhausted_path_requires_stop",
            )
        desired = path[min(1, len(path) - 1)] if path else position
        workers = np.argwhere((semantic_map[2] > 0) | (semantic_map[7] > 0))
        candidates: list[tuple[float, int, tuple[Position, ...], float]] = []
        # The action mask is expressed in sensor/grid directions (forward,
        # reverse, left, right, stop), while the controller output is the
        # canonical primitive enum.  Do not index the mask with the primitive
        # id: that was the source of forward requests becoming STOP/INSPECT.
        mask_for_primitive = {
            1: bool(mask[0]),
            2: bool(mask[2]),
            3: bool(mask[3]),
            0: bool(mask[4]),
        }
        desired_delta = (
            (
                path[1][0] - position[0],
                path[1][1] - position[1],
            )
            if len(path) > 1
            else (0, 0)
        )
        # Reverse grid progress is handled as a bounded heading correction:
        # turn toward the open side rather than issuing an unsupported reverse
        # wheel primitive.  This keeps the canonical primitive enum intact.
        if desired_delta == (1, 0):
            desired_delta = (0, 1)
        motion_choices = list(GRID_MOTION)
        if inspection_intent and len(path) <= 1:
            motion_choices.append((4, (0, 0)))
        for primitive, delta in motion_choices:
            if primitive != 4 and not mask_for_primitive[primitive]:
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
            # Prefer the primitive that realizes the next path delta.  A
            # forward path step must remain MOVE_FORWARD; lateral steps map to
            # opposite physical arcs and never to STOP/INSPECT.
            if desired_delta == delta:
                progress -= 10.0
            if primitive == 0 and desired_delta != (0, 0):
                progress += 10.0
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
            if inspection_intent and len(path) <= 1 and primitive == 4:
                # Reaching a recognized target changes semantic exposure from an
                # uncontrolled navigation cost into a shield-validated inspection.
                risk = 0.0
                progress -= 10.0
            if not inspection_intent and len(path) <= 1 and primitive == 4:
                progress += 2.0
            continuity = (
                self.continuity_weight
                if self._previous_primitive is not None and primitive != self._previous_primitive
                else 0.0
            )
            semantic_risk_weight = 0.25 if inspection_intent else 1.0
            candidates.append(
                (
                    hard + float(progress) + semantic_risk_weight * risk + continuity,
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
