from __future__ import annotations

import heapq
from dataclasses import dataclass
from itertools import count

from riskaware_saferrl.envs import ResearchConstructionEnv

Position = tuple[int, int]


@dataclass(frozen=True)
class PlannerDecision:
    action: int
    target: Position | None
    path: tuple[Position, ...]
    reason: str
    replanned: bool = True


class GridPlanner:
    """Common deterministic planning behavior."""

    name = "grid_planner"

    def __init__(self, *, risk_weight: float = 5.0) -> None:
        self.risk_weight = risk_weight
        self.known_hazards: set[Position] = set()

    def reset(self) -> None:
        self.known_hazards.clear()

    def _observe(self, environment: ResearchConstructionEnv) -> None:
        self.known_hazards.update(
            position
            for position in environment.hazards
            if environment._visible(position)  # noqa: SLF001 - planner observation adapter
        )

    def _risk(self, environment: ResearchConstructionEnv, position: Position) -> float:
        risk = 0.0
        if position in environment.restricted:
            risk += 10.0
        if position in environment.ppe_risk:
            risk += 2.0
        if environment._near(  # noqa: SLF001 - planner model uses benchmark dynamics
            position, environment.workers | environment.dynamic_hazards
        ):
            risk += 5.0
        return risk

    def _path(
        self,
        environment: ResearchConstructionEnv,
        target: Position,
    ) -> list[Position] | None:
        start = environment.agent
        if start == target:
            return [start]
        queue: list[tuple[float, float, int, Position]] = []
        order = count()
        heapq.heappush(queue, (0.0, 0.0, next(order), start))
        came_from: dict[Position, Position] = {}
        best_cost = {start: 0.0}
        while queue:
            _, current_cost, _, current = heapq.heappop(queue)
            if current == target:
                path = [current]
                while current in came_from:
                    current = came_from[current]
                    path.append(current)
                return list(reversed(path))
            if current_cost > best_cost[current]:
                continue
            for action in sorted(environment.ACTION_TO_DELTA):
                delta = environment.ACTION_TO_DELTA[action]
                candidate = current[0] + delta[0], current[1] + delta[1]
                if not environment._inside(candidate) or candidate in environment.obstacles:  # noqa: SLF001
                    continue
                candidate_cost = (
                    current_cost + 1.0 + self.risk_weight * self._risk(environment, candidate)
                )
                if candidate_cost >= best_cost.get(candidate, float("inf")):
                    continue
                best_cost[candidate] = candidate_cost
                came_from[candidate] = current
                heuristic = abs(candidate[0] - target[0]) + abs(candidate[1] - target[1])
                heapq.heappush(
                    queue,
                    (candidate_cost + heuristic, candidate_cost, next(order), candidate),
                )
        return None

    def _path_to_any(
        self,
        environment: ResearchConstructionEnv,
        targets: set[Position],
    ) -> tuple[Position, list[Position]] | None:
        """Run one deterministic uniform-cost search to the safest target."""
        start = environment.agent
        if start in targets:
            return start, [start]
        queue: list[tuple[float, int, Position]] = []
        order = count()
        heapq.heappush(queue, (0.0, next(order), start))
        came_from: dict[Position, Position] = {}
        best_cost = {start: 0.0}
        while queue:
            current_cost, _, current = heapq.heappop(queue)
            if current in targets:
                path = [current]
                while current in came_from:
                    current = came_from[current]
                    path.append(current)
                return path[0], list(reversed(path))
            if current_cost > best_cost[current]:
                continue
            for action in sorted(environment.ACTION_TO_DELTA):
                delta = environment.ACTION_TO_DELTA[action]
                candidate = current[0] + delta[0], current[1] + delta[1]
                if not environment._inside(candidate) or candidate in environment.obstacles:  # noqa: SLF001
                    continue
                candidate_cost = (
                    current_cost + 1.0 + self.risk_weight * self._risk(environment, candidate)
                )
                if candidate_cost >= best_cost.get(candidate, float("inf")):
                    continue
                best_cost[candidate] = candidate_cost
                came_from[candidate] = current
                heapq.heappush(queue, (candidate_cost, next(order), candidate))
        return None

    @staticmethod
    def _action_from_path(environment: ResearchConstructionEnv, path: list[Position]) -> int:
        if len(path) < 2:
            return 4
        delta = path[1][0] - path[0][0], path[1][1] - path[0][1]
        for action, candidate_delta in environment.ACTION_TO_DELTA.items():
            if candidate_delta == delta:
                return action
        raise RuntimeError(f"Planner produced a non-adjacent transition: {path[:2]}")

    def decide(self, environment: ResearchConstructionEnv) -> PlannerDecision:
        raise NotImplementedError


class RiskAwareAStarPlanner(GridPlanner):
    """Privileged expert that plans safe inspection viewpoints using map truth."""

    name = "risk_aware_astar"

    def decide(self, environment: ResearchConstructionEnv) -> PlannerDecision:
        if environment.inspectable_hazards():
            return PlannerDecision(4, environment.agent, (environment.agent,), "inspect")
        viewpoints: set[Position] = set()
        for hazard in sorted(environment.hazards - environment.inspected):
            for row in range(environment.size):
                for column in range(environment.size):
                    viewpoint = row, column
                    if (
                        abs(row - hazard[0]) + abs(column - hazard[1])
                        > environment.config.inspection_radius
                    ):
                        continue
                    if viewpoint not in environment.obstacles:
                        viewpoints.add(viewpoint)
        result = self._path_to_any(environment, viewpoints)
        if result is None:
            return PlannerDecision(4, None, (environment.agent,), "no_reachable_target")
        target, path = result
        return PlannerDecision(
            self._action_from_path(environment, path),
            target,
            tuple(path),
            "risk_aware_shortest_path",
        )


class FrontierExplorationPlanner(GridPlanner):
    """Select the nearest safe unvisited frontier and replan after every step."""

    name = "frontier_exploration"

    def decide(self, environment: ResearchConstructionEnv) -> PlannerDecision:
        self._observe(environment)
        if environment.inspectable_hazards():
            return PlannerDecision(4, environment.agent, (environment.agent,), "inspect")
        frontiers: set[Position] = set()
        for row in range(environment.size):
            for column in range(environment.size):
                target = row, column
                if target in environment.visited or target in environment.obstacles:
                    continue
                if not any(
                    ((target[0] + delta[0], target[1] + delta[1]) in environment.visited)
                    for delta in environment.ACTION_TO_DELTA.values()
                ):
                    continue
                frontiers.add(target)
        if not frontiers:
            return NearestRiskRevisitPlanner(risk_weight=self.risk_weight).decide(environment)
        result = self._path_to_any(environment, frontiers)
        if result is None:
            return PlannerDecision(4, None, (environment.agent,), "no_reachable_frontier")
        target, path = result
        return PlannerDecision(
            self._action_from_path(environment, path), target, tuple(path), "nearest_frontier"
        )


class NearestRiskRevisitPlanner(GridPlanner):
    """Revisit the nearest known uninspected risk location."""

    name = "nearest_risk_revisit"

    def decide(self, environment: ResearchConstructionEnv) -> PlannerDecision:
        self._observe(environment)
        if environment.inspectable_hazards():
            return PlannerDecision(4, environment.agent, (environment.agent,), "inspect")
        targets = sorted(self.known_hazards - environment.inspected)
        result = self._path_to_any(environment, set(targets)) if targets else None
        if result is None:
            unexplored = {
                (row, column)
                for row in range(environment.size)
                for column in range(environment.size)
                if (row, column) not in environment.visited
                and (row, column) not in environment.obstacles
            }
            result = self._path_to_any(environment, unexplored)
        if result is None:
            return PlannerDecision(4, None, (environment.agent,), "no_reachable_target")
        target, path = result
        return PlannerDecision(
            self._action_from_path(environment, path), target, tuple(path), "nearest_known_risk"
        )
