from __future__ import annotations

from collections.abc import Collection, Iterable

Position = tuple[int, int]


def manhattan_distance(first: Position, second: Position) -> int:
    return abs(first[0] - second[0]) + abs(first[1] - second[1])


def _bresenham_cells(
    start: Position,
    end: Position,
) -> tuple[Position, ...]:
    row, column = start
    end_row, end_column = end

    row_distance = abs(end_row - row)
    column_distance = abs(end_column - column)

    row_step = 0 if row == end_row else (1 if row < end_row else -1)
    column_step = 0 if column == end_column else (1 if column < end_column else -1)

    error = row_distance - column_distance
    cells: list[Position] = []

    while True:
        cells.append((row, column))

        if (row, column) == (end_row, end_column):
            break

        doubled_error = 2 * error

        if doubled_error > -column_distance:
            error -= column_distance
            row += row_step

        if doubled_error < row_distance:
            error += row_distance
            column += column_step

    return tuple(cells)


def grid_line_cells(
    start: Position,
    end: Position,
) -> tuple[Position, ...]:
    forward_cells = set(_bresenham_cells(start, end))
    backward_cells = set(_bresenham_cells(end, start))
    supercover_cells = forward_cells | backward_cells

    row_delta = end[0] - start[0]
    column_delta = end[1] - start[1]

    def ordering_key(position: Position) -> tuple[int, int, int]:
        projection = (position[0] - start[0]) * row_delta + (position[1] - start[1]) * column_delta

        return projection, position[0], position[1]

    return tuple(
        sorted(
            supercover_cells,
            key=ordering_key,
        )
    )


def has_line_of_sight(
    start: Position,
    target: Position,
    blockers: Collection[Position],
) -> bool:
    intermediate_cells = grid_line_cells(start, target)[1:-1]
    return all(cell not in blockers for cell in intermediate_cells)


def inspectable_hazards_from(
    viewpoint: Position,
    hazards: Iterable[Position],
    *,
    blockers: Collection[Position],
    inspection_radius: int,
) -> frozenset[Position]:
    if inspection_radius < 0:
        raise ValueError("inspection_radius must be non-negative.")

    return frozenset(
        hazard
        for hazard in hazards
        if manhattan_distance(viewpoint, hazard) <= inspection_radius
        and has_line_of_sight(viewpoint, hazard, blockers)
    )


def candidate_viewpoints(
    hazards: Iterable[Position],
    *,
    grid_size: int,
    inspection_radius: int,
) -> tuple[Position, ...]:
    if inspection_radius < 0:
        raise ValueError("inspection_radius must be non-negative.")

    candidates: set[Position] = set()

    for hazard_row, hazard_column in hazards:
        for row in range(
            max(0, hazard_row - inspection_radius),
            min(grid_size, hazard_row + inspection_radius + 1),
        ):
            remaining_distance = inspection_radius - abs(row - hazard_row)

            for column in range(
                max(0, hazard_column - remaining_distance),
                min(grid_size, hazard_column + remaining_distance + 1),
            ):
                candidates.add((row, column))

    return tuple(sorted(candidates))
