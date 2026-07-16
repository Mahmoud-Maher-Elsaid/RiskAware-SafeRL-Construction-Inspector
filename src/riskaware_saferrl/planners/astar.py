from __future__ import annotations

import heapq
from collections.abc import Collection, Sequence
from itertools import count

Position = tuple[int, int]

ACTION_TO_DELTA = {
    0: (-1, 0),
    1: (1, 0),
    2: (0, -1),
    3: (0, 1),
}

DELTA_TO_ACTION = {delta: action for action, delta in ACTION_TO_DELTA.items()}


def manhattan_distance(first: Position, second: Position) -> int:
    return abs(first[0] - second[0]) + abs(first[1] - second[1])


def neighbors(
    position: Position,
    *,
    grid_size: int,
    blocked: Collection[Position],
) -> tuple[Position, ...]:
    candidates: list[Position] = []

    for row_delta, column_delta in ACTION_TO_DELTA.values():
        candidate = (
            position[0] + row_delta,
            position[1] + column_delta,
        )

        if not (0 <= candidate[0] < grid_size and 0 <= candidate[1] < grid_size):
            continue

        if candidate in blocked:
            continue

        candidates.append(candidate)

    return tuple(candidates)


def reconstruct_path(
    came_from: dict[Position, Position],
    current: Position,
) -> list[Position]:
    path = [current]

    while current in came_from:
        current = came_from[current]
        path.append(current)

    path.reverse()
    return path


def astar_path(
    start: Position,
    goal: Position,
    *,
    grid_size: int,
    blocked: Collection[Position],
) -> list[Position] | None:
    if start == goal:
        return [start]

    if goal in blocked:
        return None

    queue: list[tuple[int, int, int, Position]] = []
    sequence = count()

    heapq.heappush(
        queue,
        (
            manhattan_distance(start, goal),
            0,
            next(sequence),
            start,
        ),
    )

    came_from: dict[Position, Position] = {}
    best_cost = {start: 0}

    while queue:
        _, current_cost, _, current = heapq.heappop(queue)

        if current == goal:
            return reconstruct_path(came_from, current)

        if current_cost > best_cost[current]:
            continue

        for candidate in neighbors(
            current,
            grid_size=grid_size,
            blocked=blocked,
        ):
            candidate_cost = current_cost + 1

            if candidate_cost >= best_cost.get(candidate, 10**12):
                continue

            best_cost[candidate] = candidate_cost
            came_from[candidate] = current

            priority = candidate_cost + manhattan_distance(candidate, goal)

            heapq.heappush(
                queue,
                (
                    priority,
                    candidate_cost,
                    next(sequence),
                    candidate,
                ),
            )

    return None


def path_to_actions(path: Sequence[Position]) -> list[int]:
    if len(path) < 2:
        return []

    actions: list[int] = []

    for current, following in zip(path, path[1:], strict=False):
        delta = (
            following[0] - current[0],
            following[1] - current[1],
        )

        if delta not in DELTA_TO_ACTION:
            raise ValueError(f"Path contains a non-adjacent transition: {current} -> {following}")

        actions.append(DELTA_TO_ACTION[delta])

    return actions
