from __future__ import annotations

import heapq
from dataclasses import dataclass
from itertools import count

import numpy as np
from numpy.typing import NDArray

Position = tuple[int, int]
ACTION_DELTAS: dict[int, Position] = {
    0: (-1, 0),
    1: (1, 0),
    2: (0, -1),
    3: (0, 1),
}


@dataclass(frozen=True)
class CausalExpertDecision:
    proposed_action: int
    final_action: int
    shield_decision: str
    target_type: str
    target: Position | None
    reason: str
    predicted_observed_cost: float


class CausalObservationExpert:
    """Deterministic planner restricted to policy observations and derived memory."""

    def __init__(
        self,
        *,
        inspection_radius: int = 2,
        risk_weight: float = 5.0,
        shield_threshold: float = 1.0,
    ) -> None:
        self.inspection_radius = inspection_radius
        self.risk_weight = risk_weight
        self.shield_threshold = shield_threshold
        self.reset()

    def reset(self) -> None:
        self._observed = np.zeros((16, 16), dtype=np.bool_)
        self._known_free = np.zeros((16, 16), dtype=np.bool_)
        self._known_obstacles = np.zeros((16, 16), dtype=np.bool_)
        self._known_hazards = np.zeros((16, 16), dtype=np.bool_)
        self._inspected = np.zeros((16, 16), dtype=np.bool_)
        self._visited = np.zeros((16, 16), dtype=np.bool_)
        self._visit_counts = np.zeros((16, 16), dtype=np.int32)
        self._risk = np.zeros((16, 16), dtype=np.float32)
        self._recent_positions: list[Position] = []
        self._step = 0

    @property
    def discovered_target_count(self) -> int:
        return int(np.count_nonzero(self._known_hazards & ~self._inspected))

    def memory_snapshot(self) -> dict[str, NDArray[np.generic] | int]:
        return {
            "observed": self._observed.copy(),
            "known_free": self._known_free.copy(),
            "known_obstacles": self._known_obstacles.copy(),
            "known_hazards": self._known_hazards.copy(),
            "inspected": self._inspected.copy(),
            "visited": self._visited.copy(),
            "visit_counts": self._visit_counts.copy(),
            "risk": self._risk.copy(),
            "step": self._step,
        }

    @staticmethod
    def _validate_observation(observation: dict[str, NDArray[np.generic]]) -> None:
        required = {"map", "state", "action_mask"}
        if set(observation) != required:
            raise ValueError(f"Observation keys must be exactly {sorted(required)}")
        if observation["map"].shape != (11, 16, 16):
            raise ValueError("Expected semantic map shape (11, 16, 16)")
        if observation["state"].shape != (17,):
            raise ValueError("Expected state shape (17,)")
        if observation["action_mask"].shape != (5,):
            raise ValueError("Expected action-mask shape (5,)")

    def _update_memory(self, observation: dict[str, NDArray[np.generic]]) -> Position:
        self._validate_observation(observation)
        semantic_map = np.asarray(observation["map"])
        visible = semantic_map[9] > 0
        obstacles = semantic_map[0] > 0
        self._observed |= visible | obstacles
        self._known_obstacles |= obstacles
        self._known_free |= visible & ~obstacles
        self._known_hazards |= semantic_map[1] > 0
        self._inspected |= semantic_map[10] > 0
        self._visited |= semantic_map[4] > 0
        persistent_risk = np.maximum.reduce(
            (
                (semantic_map[1] > 0).astype(np.float32) * 0.7,
                (semantic_map[3] > 0).astype(np.float32),
                (semantic_map[8] > 0).astype(np.float32) * 0.65,
            )
        )
        current_risk = np.maximum.reduce(
            (
                semantic_map[6].astype(np.float32),
                (semantic_map[2] > 0).astype(np.float32) * 0.9,
                (semantic_map[7] > 0).astype(np.float32) * 0.85,
            )
        )
        self._risk = np.maximum(persistent_risk, current_risk)
        locations = np.argwhere(semantic_map[5] > 0)
        if len(locations) != 1:
            raise ValueError("Observation must contain exactly one robot position")
        position = int(locations[0, 0]), int(locations[0, 1])
        self._known_free[position] = True
        self._visited[position] = True
        self._visit_counts[position] += 1
        action_mask = np.asarray(observation["action_mask"], dtype=np.bool_)
        for action, delta in ACTION_DELTAS.items():
            candidate = position[0] + delta[0], position[1] + delta[1]
            if not self._inside(candidate, 16):
                continue
            if action_mask[action]:
                self._known_free[candidate] = True
            else:
                self._known_obstacles[candidate] = True
                self._known_free[candidate] = False
        self._recent_positions.append(position)
        self._recent_positions = self._recent_positions[-12:]
        self._step += 1
        return position

    def _inside(self, position: Position, size: int) -> bool:
        return 0 <= position[0] < size and 0 <= position[1] < size

    def _observed_action_cost(self, position: Position, action: int, size: int) -> float:
        if action == 4:
            return 0.0
        delta = ACTION_DELTAS[action]
        candidate = position[0] + delta[0], position[1] + delta[1]
        if not self._inside(candidate, size) or self._known_obstacles[candidate]:
            return float("inf")
        return float(self._risk[candidate])

    def _path_to_any(
        self,
        start: Position,
        targets: set[Position],
        size: int,
    ) -> tuple[Position, int] | None:
        if not targets:
            return None
        queue: list[tuple[float, int, Position, int | None]] = []
        order = count()
        heapq.heappush(queue, (0.0, next(order), start, None))
        best = {start: 0.0}
        while queue:
            cost, _, position, first_action = heapq.heappop(queue)
            if cost != best.get(position):
                continue
            if position in targets:
                return position, 4 if first_action is None else first_action
            for action, delta in ACTION_DELTAS.items():
                candidate = position[0] + delta[0], position[1] + delta[1]
                if (
                    not self._inside(candidate, size)
                    or not self._known_free[candidate]
                    or self._known_obstacles[candidate]
                ):
                    continue
                candidate_cost = cost + 1.0 + self.risk_weight * float(self._risk[candidate])
                if candidate_cost >= best.get(candidate, float("inf")):
                    continue
                best[candidate] = candidate_cost
                heapq.heappush(
                    queue,
                    (
                        candidate_cost,
                        next(order),
                        candidate,
                        action if first_action is None else first_action,
                    ),
                )
        return None

    def _visible_target_viewpoints(self, size: int) -> set[Position]:
        hazards = np.argwhere(self._known_hazards & ~self._inspected)
        viewpoints: set[Position] = set()
        for hazard_row, hazard_column in hazards:
            for row in range(size):
                for column in range(size):
                    candidate = row, column
                    if (
                        abs(row - int(hazard_row)) + abs(column - int(hazard_column))
                        <= self.inspection_radius
                        and self._known_free[candidate]
                        and not self._known_obstacles[candidate]
                    ):
                        viewpoints.add(candidate)
        return viewpoints

    def _frontiers(self, size: int) -> set[Position]:
        frontiers: set[Position] = set()
        for row in range(size):
            for column in range(size):
                position = row, column
                if not self._known_free[position]:
                    continue
                if any(
                    self._inside((row + dr, column + dc), size)
                    and not self._observed[row + dr, column + dc]
                    for dr, dc in ACTION_DELTAS.values()
                ):
                    frontiers.add(position)
        return frontiers

    def _exploration_action(
        self,
        position: Position,
        mask: NDArray[np.bool_],
        size: int,
    ) -> int:
        candidates: list[tuple[int, float, int]] = []
        recent = set(self._recent_positions[-6:])
        for action, delta in ACTION_DELTAS.items():
            if not mask[action]:
                continue
            candidate = position[0] + delta[0], position[1] + delta[1]
            unknown_priority = 0 if not self._observed[candidate] else 1
            revisit_penalty = float(self._visit_counts[candidate]) + (
                2.0 if candidate in recent else 0.0
            )
            candidates.append(
                (
                    unknown_priority,
                    float(self._risk[candidate]) + revisit_penalty,
                    action,
                )
            )
        if candidates:
            return min(candidates)[2]
        return int(np.flatnonzero(mask)[0])

    def _shield(
        self,
        proposed: int,
        position: Position,
        mask: NDArray[np.bool_],
        size: int,
    ) -> tuple[int, str, float]:
        proposed_cost = self._observed_action_cost(position, proposed, size)
        if mask[proposed] and proposed_cost <= self.shield_threshold:
            return proposed, "accept", proposed_cost
        candidates = [
            (self._observed_action_cost(position, action, size), action)
            for action in range(5)
            if mask[action]
        ]
        if not candidates:
            raise ValueError("Action mask contains no valid action")
        cost, replacement = min(candidates)
        return replacement, "replace", float(cost)

    def decide(self, observation: dict[str, NDArray[np.generic]]) -> CausalExpertDecision:
        position = self._update_memory(observation)
        state = np.asarray(observation["state"])
        size = int(round(float(state[9]) * 16.0))
        size = max(1, min(16, size))
        mask = np.asarray(observation["action_mask"], dtype=np.bool_)
        if not np.any(mask):
            raise ValueError("Action mask contains no valid action")

        remaining_targets = self._known_hazards & ~self._inspected
        target_type = "frontier"
        target: Position | None = None
        reason = "explore_unknown"
        if mask[4]:
            proposed = 4
            target_type = "observed_risk" if np.any(remaining_targets) else "action_mask_risk"
            target = position
            reason = (
                "inspect_observed_target"
                if np.any(remaining_targets)
                else "inspect_mask_permitted_target"
            )
        else:
            result = self._path_to_any(position, self._visible_target_viewpoints(size), size)
            if result is not None and result[1] != 4:
                target, proposed = result
                target_type = "observed_risk"
                reason = "navigate_to_observed_target"
            else:
                frontier_result = self._path_to_any(position, self._frontiers(size), size)
                if frontier_result is not None and frontier_result[1] != 4:
                    target, proposed = frontier_result
                    reason = "navigate_to_frontier"
                else:
                    proposed = self._exploration_action(position, mask, size)
                    reason = "expand_discovered_map"
        if not mask[proposed]:
            proposed = int(np.flatnonzero(mask)[0])
            reason = "valid_mask_fallback"
        final, shield_decision, predicted_cost = self._shield(proposed, position, mask, size)
        return CausalExpertDecision(
            proposed_action=int(proposed),
            final_action=int(final),
            shield_decision=shield_decision,
            target_type=target_type,
            target=target,
            reason=reason,
            predicted_observed_cost=predicted_cost,
        )
