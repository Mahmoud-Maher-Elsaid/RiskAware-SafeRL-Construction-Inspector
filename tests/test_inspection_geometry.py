from riskaware_saferrl.inspection import (
    grid_line_cells,
    has_line_of_sight,
    inspectable_hazards_from,
)


def test_grid_line_uses_symmetric_supercover_cells() -> None:
    forward = grid_line_cells((0, 0), (2, 1))
    backward = grid_line_cells((2, 1), (0, 0))

    expected_cells = frozenset(
        {
            (0, 0),
            (1, 0),
            (1, 1),
            (2, 1),
        }
    )

    assert frozenset(forward) == expected_cells
    assert frozenset(backward) == expected_cells
    assert forward[0] == (0, 0)
    assert forward[-1] == (2, 1)
    assert backward[0] == (2, 1)
    assert backward[-1] == (0, 0)


def test_line_of_sight_is_symmetric() -> None:
    blockers = {(1, 0)}

    assert not has_line_of_sight(
        (0, 0),
        (2, 1),
        blockers,
    )
    assert not has_line_of_sight(
        (2, 1),
        (0, 0),
        blockers,
    )


def test_line_of_sight_is_blocked_by_intermediate_obstacle() -> None:
    assert not has_line_of_sight(
        (0, 0),
        (0, 2),
        {(0, 1)},
    )


def test_inspectable_hazards_respect_radius_and_visibility() -> None:
    hazards = {
        (0, 2),
        (2, 0),
        (3, 0),
    }

    visible = inspectable_hazards_from(
        (0, 0),
        hazards,
        blockers={(0, 1)},
        inspection_radius=2,
    )

    assert visible == frozenset({(2, 0)})
