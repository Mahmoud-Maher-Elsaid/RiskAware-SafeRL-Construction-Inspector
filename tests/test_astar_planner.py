from riskaware_saferrl.planners import (
    astar_path,
    path_to_actions,
)


def test_astar_finds_shortest_open_path() -> None:
    path = astar_path(
        (0, 0),
        (0, 3),
        grid_size=5,
        blocked=set(),
    )

    assert path == [
        (0, 0),
        (0, 1),
        (0, 2),
        (0, 3),
    ]
    assert path_to_actions(path) == [3, 3, 3]


def test_astar_routes_around_obstacle() -> None:
    path = astar_path(
        (0, 0),
        (0, 2),
        grid_size=4,
        blocked={(0, 1)},
    )

    assert path is not None
    assert (0, 1) not in path
    assert len(path) == 5


def test_astar_returns_none_for_blocked_goal() -> None:
    path = astar_path(
        (0, 0),
        (1, 1),
        grid_size=3,
        blocked={(1, 1)},
    )

    assert path is None
