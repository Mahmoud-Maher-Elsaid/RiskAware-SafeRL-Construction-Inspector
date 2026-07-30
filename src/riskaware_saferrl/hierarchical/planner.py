from __future__ import annotations

import heapq
from dataclasses import dataclass
from itertools import count

import numpy as np

from riskaware_saferrl.hierarchical.schemas import MissionOption

Position = tuple[int, int]
DELTAS: tuple[Position, ...] = ((-1, 0), (1, 0), (0, -1), (0, 1))


@dataclass(frozen=True)
class PlannerRequest:
    option: MissionOption
    target: Position | None
    risk_budget: float
    inspection_intent: bool
    force_replan: bool = False


@dataclass(frozen=True)
class PlannerResult:
    path: tuple[Position, ...]
    target: Position | None
    target_type: str
    success: bool
    replanned: bool
    reason: str
    expanded_nodes: int
    observed_map_revision: int


class CausalRiskAwarePlanner:
    """Incremental risk-aware A* using only persistent observation-derived maps."""

    def __init__(self, *, size: int = 16) -> None:
        self.size = size
        self.reset()

    def reset(self) -> None:
        shape = (self.size, self.size)
        self._observed = np.zeros(shape, dtype=np.bool_)
        self._free = np.zeros(shape, dtype=np.bool_)
        self._obstacle = np.zeros(shape, dtype=np.bool_)
        self._restricted = np.zeros(shape, dtype=np.bool_)
        self._worker = np.zeros(shape, dtype=np.float32)
        self._risk = np.zeros(shape, dtype=np.float32)
        self._uncertainty = np.ones(shape, dtype=np.float32)
        self._visited = np.zeros(shape, dtype=np.int32)
        self._revision = 0
        self._last_signature: bytes | None = None
        self._last_path: tuple[Position, ...] = ()
        self._last_target: Position | None = None

    @staticmethod
    def _robot_position(semantic_map: np.ndarray) -> Position:
        locations = np.argwhere(semantic_map[5] > 0)
        if len(locations) != 1:
            raise ValueError("Observation must contain exactly one robot cell")
        return int(locations[0, 0]), int(locations[0, 1])

    def _inside(self, position: Position) -> bool:
        return 0 <= position[0] < self.size and 0 <= position[1] < self.size

    def update(self, observation: dict[str, np.ndarray]) -> Position:
        if set(observation) != {"map", "state", "action_mask"}:
            raise ValueError("Planner accepts only map, state, and action_mask observations")
        semantic_map = np.asarray(observation["map"], dtype=np.float32)
        if semantic_map.shape != (11, self.size, self.size):
            raise ValueError("Unexpected semantic-map shape")
        visible = semantic_map[9] > 0
        obstacle = semantic_map[0] > 0
        self._observed |= visible | obstacle
        self._obstacle |= obstacle
        self._free |= visible & ~obstacle
        self._free[self._obstacle] = False
        self._restricted = np.where(visible, semantic_map[3] > 0, self._restricted)
        self._worker = np.where(visible, semantic_map[2], self._worker * 0.8)
        observed_risk = np.maximum.reduce((semantic_map[1], semantic_map[6], semantic_map[8]))
        self._risk = np.where(visible, observed_risk, self._risk)
        self._uncertainty = np.where(visible, 0.0, 1.0)
        position = self._robot_position(semantic_map)
        self._free[position] = True
        self._visited[position] += 1
        signature = np.packbits(
            np.stack((self._free, self._obstacle, self._restricted, self._worker > 0))
        ).tobytes()
        if signature != self._last_signature:
            self._revision += 1
            self._last_signature = signature
        return position

    def _frontiers(self) -> set[Position]:
        values: set[Position] = set()
        for row, column in np.argwhere(self._free):
            position = int(row), int(column)
            if any(
                self._inside((position[0] + dr, position[1] + dc))
                and not self._observed[position[0] + dr, position[1] + dc]
                for dr, dc in DELTAS
            ):
                values.add(position)
        return values

    def _safe_cells(self) -> set[Position]:
        return {
            (int(row), int(column))
            for row, column in np.argwhere(
                self._free & ~self._restricted & (self._worker <= 0) & (self._risk < 0.25)
            )
        }

    def _observed_targets(self, semantic_map: np.ndarray, ppe_only: bool) -> set[Position]:
        channel = semantic_map[8] if ppe_only else np.maximum(semantic_map[1], semantic_map[8])
        targets = set()
        for row, column in np.argwhere(channel > 0):
            target = int(row), int(column)
            for dr, dc in DELTAS + ((0, 0),):
                candidate = target[0] + dr, target[1] + dc
                if self._inside(candidate) and self._free[candidate]:
                    targets.add(candidate)
        return targets

    def _step_cost(self, previous: Position, candidate: Position, risk_budget: float) -> float:
        if self._obstacle[candidate] or self._restricted[candidate]:
            return float("inf")
        clearance = min(
            (
                abs(candidate[0] - int(row)) + abs(candidate[1] - int(column))
                for row, column in np.argwhere(self._worker > 0)
            ),
            default=99,
        )
        if clearance < 1:
            return float("inf")
        return (
            1.0
            + 2.0 * float(self._visited[candidate])
            + (6.0 - 4.0 * risk_budget) * float(self._risk[candidate])
            + 3.0 * float(self._uncertainty[candidate])
            + (2.0 if clearance == 1 else 0.0)
            + (0.1 if previous == candidate else 0.0)
        )

    def _search(
        self, start: Position, targets: set[Position], risk_budget: float
    ) -> tuple[tuple[Position, ...], int]:
        if not targets:
            return (), 0
        queue: list[tuple[float, int, Position]] = [(0.0, 0, start)]
        order = count(1)
        cost = {start: 0.0}
        parent: dict[Position, Position] = {}
        expanded = 0
        reached: Position | None = None
        while queue:
            _, _, current = heapq.heappop(queue)
            expanded += 1
            if current in targets:
                reached = current
                break
            for delta in DELTAS:
                candidate = current[0] + delta[0], current[1] + delta[1]
                if not self._inside(candidate) or not self._free[candidate]:
                    continue
                new_cost = cost[current] + self._step_cost(current, candidate, risk_budget)
                if new_cost >= cost.get(candidate, float("inf")):
                    continue
                cost[candidate] = new_cost
                parent[candidate] = current
                heuristic = min(
                    abs(candidate[0] - target[0]) + abs(candidate[1] - target[1])
                    for target in targets
                )
                heapq.heappush(queue, (new_cost + heuristic, next(order), candidate))
        if reached is None:
            return (), expanded
        path = [reached]
        while path[-1] != start:
            path.append(parent[path[-1]])
        return tuple(reversed(path)), expanded

    def plan(self, observation: dict[str, np.ndarray], request: PlannerRequest) -> PlannerResult:
        semantic_map = np.asarray(observation["map"])
        start = self.update(observation)
        target_type = "requested"
        targets: set[Position]
        if request.option == MissionOption.EXPLORE_FRONTIER:
            targets = self._frontiers()
            target_type = "frontier"
        elif request.option == MissionOption.INSPECT_PPE_VIOLATION:
            targets = self._observed_targets(semantic_map, True)
            target_type = "observed_ppe_risk"
        elif request.option in {
            MissionOption.INSPECT_KNOWN_RISK,
            MissionOption.CONTINUE_CURRENT_TARGET,
        }:
            targets = (
                {request.target}
                if request.target is not None
                else self._observed_targets(semantic_map, False)
            )
            target_type = "observed_risk"
        elif request.option in {
            MissionOption.AVOID_DYNAMIC_WORKER,
            MissionOption.RETREAT_TO_SAFE_CELL,
        }:
            targets = self._safe_cells()
            target_type = "safe_retreat"
        elif request.option in {
            MissionOption.HOLD_FOR_UNCERTAINTY,
            MissionOption.EMERGENCY_SAFE_STOP,
        }:
            return PlannerResult(
                path=(start,),
                target=start,
                target_type="hold",
                success=True,
                replanned=False,
                reason="option_requests_hold",
                expanded_nodes=0,
                observed_map_revision=self._revision,
            )
        else:
            targets = self._frontiers() or self._safe_cells()
            target_type = "replan"
        can_reuse = (
            not request.force_replan
            and self._last_path
            and self._last_target in targets
            and start in self._last_path
        )
        if can_reuse:
            path = self._last_path[self._last_path.index(start) :]
            expanded = 0
            replanned = False
        else:
            path, expanded = self._search(start, targets, request.risk_budget)
            replanned = True
        if not path:
            return PlannerResult(
                path=(start,),
                target=None,
                target_type=target_type,
                success=False,
                replanned=replanned,
                reason="no_causal_route",
                expanded_nodes=expanded,
                observed_map_revision=self._revision,
            )
        self._last_path = path
        self._last_target = path[-1]
        return PlannerResult(
            path=path,
            target=path[-1],
            target_type=target_type,
            success=True,
            replanned=replanned,
            reason="causal_route_found",
            expanded_nodes=expanded,
            observed_map_revision=self._revision,
        )
