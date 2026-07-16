from __future__ import annotations

from dataclasses import dataclass

from riskaware_saferrl.planners.astar import (
    Position,
    astar_path,
    path_to_actions,
)
from riskaware_saferrl.scenarios import Scenario


@dataclass(frozen=True)
class InspectionPlan:
    scenario_id: str
    planner_name: str
    actions: tuple[int, ...]
    visited_hazards: tuple[Position, ...]
    unreachable_hazards: tuple[Position, ...]
    safety_blocked_cells: tuple[Position, ...]

    @property
    def complete(self) -> bool:
        return not self.unreachable_hazards

    @property
    def movement_actions(self) -> int:
        return sum(action != 4 for action in self.actions)

    @property
    def inspection_actions(self) -> int:
        return sum(action == 4 for action in self.actions)


def worker_exclusion_cells(
    workers: tuple[Position, ...],
    *,
    grid_size: int,
) -> set[Position]:
    excluded: set[Position] = set()

    for row, column in workers:
        for row_delta, column_delta in (
            (0, 0),
            (-1, 0),
            (1, 0),
            (0, -1),
            (0, 1),
        ):
            candidate = (
                row + row_delta,
                column + column_delta,
            )

            if 0 <= candidate[0] < grid_size and 0 <= candidate[1] < grid_size:
                excluded.add(candidate)

    return excluded


def build_blocked_cells(
    scenario: Scenario,
    *,
    safety_aware: bool,
) -> set[Position]:
    blocked = set(scenario.obstacles)

    if safety_aware:
        blocked.update(scenario.restricted_zones)
        blocked.update(
            worker_exclusion_cells(
                scenario.workers,
                grid_size=scenario.grid_size,
            )
        )

    blocked.discard(scenario.agent_start)
    return blocked


def select_nearest_reachable_hazard(
    current: Position,
    remaining_hazards: set[Position],
    *,
    scenario: Scenario,
    blocked: set[Position],
) -> tuple[Position, list[Position]] | None:
    candidates: list[tuple[int, Position, list[Position]]] = []

    for hazard in sorted(remaining_hazards):
        path = astar_path(
            current,
            hazard,
            grid_size=scenario.grid_size,
            blocked=blocked,
        )

        if path is None:
            continue

        candidates.append(
            (
                len(path),
                hazard,
                path,
            )
        )

    if not candidates:
        return None

    _, hazard, path = min(
        candidates,
        key=lambda item: (
            item[0],
            item[1][0],
            item[1][1],
        ),
    )

    return hazard, path


def build_oracle_inspection_plan(
    scenario: Scenario,
    *,
    safety_aware: bool,
) -> InspectionPlan:
    scenario.validate()

    blocked = build_blocked_cells(
        scenario,
        safety_aware=safety_aware,
    )

    remaining_hazards = set(scenario.hazards)
    visited_hazards: list[Position] = []
    actions: list[int] = []
    current = scenario.agent_start

    while remaining_hazards:
        selected = select_nearest_reachable_hazard(
            current,
            remaining_hazards,
            scenario=scenario,
            blocked=blocked,
        )

        if selected is None:
            break

        hazard, path = selected
        actions.extend(path_to_actions(path))
        actions.append(4)

        current = hazard
        visited_hazards.append(hazard)
        remaining_hazards.remove(hazard)

    planner_name = "safe_greedy_astar" if safety_aware else "greedy_astar"

    return InspectionPlan(
        scenario_id=scenario.scenario_id,
        planner_name=planner_name,
        actions=tuple(actions),
        visited_hazards=tuple(visited_hazards),
        unreachable_hazards=tuple(sorted(remaining_hazards)),
        safety_blocked_cells=tuple(sorted(blocked)),
    )
