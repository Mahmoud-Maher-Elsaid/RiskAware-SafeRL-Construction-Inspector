from __future__ import annotations

import math

import numpy as np
import pytest

from riskaware_saferrl.webots import (
    ActionBridge,
    BridgeState,
    CardinalHeading,
    GridAction,
    GridFrame,
    MotionPrimitive,
    ObservationBridge,
    SemanticScene,
    WebotsSensorSnapshot,
)


def make_state(
    position: tuple[int, int],
    *,
    visited: frozenset[tuple[int, int]] | None = None,
    inspected: frozenset[tuple[int, int]] | None = None,
    steps: int = 25,
) -> BridgeState:
    return BridgeState(
        agent_position=position,
        visited=(visited if visited is not None else frozenset({position})),
        inspected=(inspected if inspected is not None else frozenset()),
        steps=steps,
        max_steps=250,
    )


def test_grid_frame_round_trip() -> None:
    frame = GridFrame()

    for row in range(frame.size):
        for column in range(frame.size):
            position = (row, column)

            x, z = frame.grid_to_world(position)

            assert (
                frame.world_to_grid(
                    x,
                    z,
                )
                == position
            )


def test_grid_frame_rejects_outside_pose() -> None:
    frame = GridFrame()

    with pytest.raises(
        ValueError,
        match="outside the grid",
    ):
        frame.world_to_grid(
            100.0,
            0.0,
        )


@pytest.mark.parametrize(
    ("yaw", "expected"),
    [
        (
            0.0,
            CardinalHeading.EAST,
        ),
        (
            math.pi / 2.0,
            CardinalHeading.NORTH,
        ),
        (
            math.pi,
            CardinalHeading.WEST,
        ),
        (
            -math.pi / 2.0,
            CardinalHeading.SOUTH,
        ),
    ],
)
def test_heading_from_yaw(
    yaw: float,
    expected: CardinalHeading,
) -> None:
    assert CardinalHeading.from_yaw(yaw) is expected


def test_sensor_snapshot_projection() -> None:
    frame = GridFrame()

    x, z = frame.grid_to_world((6, 6))

    snapshot = WebotsSensorSnapshot(
        x=x,
        z=z,
        yaw_radians=0.0,
        proximity=(
            0.0,
            4095.0,
            2047.5,
        ),
    )

    assert snapshot.grid_position(frame) == (6, 6)

    assert snapshot.heading is (CardinalHeading.EAST)

    np.testing.assert_allclose(
        snapshot.normalized_proximity(),
        np.asarray(
            [
                0.0,
                1.0,
                0.5,
            ],
            dtype=np.float32,
        ),
    )


@pytest.mark.parametrize(
    (
        "action",
        "heading",
        "expected",
    ),
    [
        (
            GridAction.MOVE_RIGHT,
            CardinalHeading.EAST,
            (MotionPrimitive.MOVE_FORWARD,),
        ),
        (
            GridAction.MOVE_UP,
            CardinalHeading.EAST,
            (
                MotionPrimitive.TURN_LEFT,
                MotionPrimitive.MOVE_FORWARD,
            ),
        ),
        (
            GridAction.MOVE_DOWN,
            CardinalHeading.EAST,
            (
                MotionPrimitive.TURN_RIGHT,
                MotionPrimitive.MOVE_FORWARD,
            ),
        ),
        (
            GridAction.MOVE_LEFT,
            CardinalHeading.EAST,
            (
                MotionPrimitive.TURN_LEFT,
                MotionPrimitive.TURN_LEFT,
                MotionPrimitive.MOVE_FORWARD,
            ),
        ),
        (
            GridAction.INSPECT,
            CardinalHeading.NORTH,
            (MotionPrimitive.INSPECT,),
        ),
    ],
)
def test_action_execution_plan(
    action: GridAction,
    heading: CardinalHeading,
    expected: tuple[
        MotionPrimitive,
        ...,
    ],
) -> None:
    bridge = ActionBridge()

    assert (
        bridge.plan(
            action,
            heading,
        )
        == expected
    )


def test_observation_contract() -> None:
    scene = SemanticScene(
        size=12,
        obstacles=frozenset(
            {
                (5, 6),
            }
        ),
        hazards=frozenset(
            {
                (6, 7),
                (0, 0),
            }
        ),
        workers=frozenset(
            {
                (7, 6),
            }
        ),
        restricted=frozenset(
            {
                (6, 5),
            }
        ),
    )

    state = make_state(
        (6, 6),
        visited=frozenset(
            {
                (6, 6),
                (6, 5),
            }
        ),
    )

    bridge = ObservationBridge(
        vision_radius=3,
        inspection_radius=2,
    )

    observation = bridge.build_observation(
        scene,
        state,
    )

    assert set(observation) == {
        "map",
        "state",
    }

    assert observation["map"].shape == (7 * 12 * 12,)

    assert observation["state"].shape == (4,)

    assert observation["map"].dtype == np.float32

    assert observation["state"].dtype == np.float32

    semantic_map = observation["map"].reshape(
        7,
        12,
        12,
    )

    assert (
        semantic_map[
            0,
            5,
            6,
        ]
        == 1.0
    )

    assert (
        semantic_map[
            1,
            6,
            7,
        ]
        == 1.0
    )

    assert (
        semantic_map[
            1,
            0,
            0,
        ]
        == 0.0
    )

    assert (
        semantic_map[
            2,
            7,
            6,
        ]
        == 1.0
    )

    assert (
        semantic_map[
            3,
            6,
            5,
        ]
        == 1.0
    )

    assert (
        semantic_map[
            4,
            6,
            5,
        ]
        == 1.0
    )

    assert (
        semantic_map[
            5,
            6,
            6,
        ]
        == 1.0
    )

    assert (
        semantic_map[
            6,
            6,
            5,
        ]
        == 1.0
    )

    assert semantic_map[
        6,
        7,
        6,
    ] == pytest.approx(0.85)


def test_action_mask_matches_task_rules() -> None:
    scene = SemanticScene(
        size=12,
        obstacles=frozenset(
            {
                (5, 6),
                (6, 5),
            }
        ),
        hazards=frozenset(
            {
                (6, 7),
            }
        ),
    )

    state = make_state((6, 6))

    bridge = ObservationBridge(inspection_radius=2)

    mask = bridge.action_mask(
        scene,
        state,
    )

    assert mask.dtype == np.bool_

    assert mask.tolist() == [
        False,
        True,
        False,
        True,
        True,
    ]


def test_boundary_action_mask() -> None:
    scene = SemanticScene(size=12)

    state = make_state((0, 0))

    mask = ObservationBridge().action_mask(
        scene,
        state,
    )

    assert mask.tolist() == [
        False,
        True,
        False,
        True,
        False,
    ]


def test_inspected_hazard_is_hidden() -> None:
    scene = SemanticScene(
        size=12,
        hazards=frozenset(
            {
                (6, 7),
            }
        ),
    )

    state = make_state(
        (6, 6),
        inspected=frozenset(
            {
                (6, 7),
            }
        ),
    )

    observation = ObservationBridge().build_observation(
        scene,
        state,
    )

    semantic_map = observation["map"].reshape(
        7,
        12,
        12,
    )

    assert (
        semantic_map[
            1,
            6,
            7,
        ]
        == 0.0
    )

    assert observation["state"][3] == 1.0


def test_obstacle_blocks_inspection_line() -> None:
    scene = SemanticScene(
        size=12,
        obstacles=frozenset(
            {
                (6, 7),
            }
        ),
        hazards=frozenset(
            {
                (6, 8),
            }
        ),
    )

    state = make_state((6, 6))

    bridge = ObservationBridge(inspection_radius=3)

    assert not bridge.inspectable_hazards(
        scene,
        state,
    )

    assert not bridge.action_mask(
        scene,
        state,
    )[int(GridAction.INSPECT)]


def test_emergency_hold_fallback() -> None:
    scene = SemanticScene(
        size=3,
        obstacles=frozenset(
            {
                (0, 1),
                (1, 0),
                (1, 2),
                (2, 1),
            }
        ),
    )

    state = BridgeState(
        agent_position=(1, 1),
        visited=frozenset(
            {
                (1, 1),
            }
        ),
        inspected=frozenset(),
        steps=0,
        max_steps=10,
    )

    mask = ObservationBridge().action_mask(
        scene,
        state,
    )

    assert mask.tolist() == [
        False,
        False,
        False,
        False,
        True,
    ]


def test_invalid_scene_position_is_rejected() -> None:
    with pytest.raises(
        ValueError,
        match="invalid position",
    ):
        SemanticScene(
            size=12,
            workers=frozenset(
                {
                    (12, 0),
                }
            ),
        )
