from __future__ import annotations

import pytest

from riskaware_saferrl.webots import (
    DifferentialDriveMapper,
    MotionPrimitive,
    WheelCommand,
)


def test_forward_command_uses_equal_positive_velocities() -> None:
    mapper = DifferentialDriveMapper(
        forward_velocity=4.0,
        turn_velocity=2.0,
    )

    assert mapper.command_for(MotionPrimitive.MOVE_FORWARD) == WheelCommand(4.0, 4.0)


@pytest.mark.parametrize(
    ("primitive", "expected"),
    [
        (
            MotionPrimitive.TURN_LEFT,
            WheelCommand(-2.0, 2.0),
        ),
        (
            MotionPrimitive.TURN_RIGHT,
            WheelCommand(2.0, -2.0),
        ),
        (
            MotionPrimitive.STOP,
            WheelCommand(0.0, 0.0),
        ),
        (
            MotionPrimitive.INSPECT,
            WheelCommand(0.0, 0.0),
        ),
    ],
)
def test_motion_mapping(
    primitive: MotionPrimitive,
    expected: WheelCommand,
) -> None:
    mapper = DifferentialDriveMapper(
        forward_velocity=4.0,
        turn_velocity=2.0,
    )

    assert mapper.command_for(primitive) == expected


def test_invalid_motion_primitive_is_rejected() -> None:
    mapper = DifferentialDriveMapper()

    with pytest.raises(
        ValueError,
        match="Unsupported motion primitive",
    ):
        mapper.command_for(999)
