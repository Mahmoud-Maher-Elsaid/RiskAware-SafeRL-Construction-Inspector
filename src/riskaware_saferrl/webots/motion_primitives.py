from __future__ import annotations

import math
from dataclasses import dataclass
from enum import IntEnum


class MotionPrimitive(IntEnum):
    STOP = 0
    MOVE_FORWARD = 1
    TURN_LEFT = 2
    TURN_RIGHT = 3
    INSPECT = 4


@dataclass(frozen=True)
class WheelCommand:
    left_velocity: float
    right_velocity: float

    def __post_init__(self) -> None:
        if not math.isfinite(self.left_velocity):
            raise ValueError("left_velocity must be finite.")
        if not math.isfinite(self.right_velocity):
            raise ValueError("right_velocity must be finite.")


@dataclass(frozen=True)
class DifferentialDriveMapper:
    forward_velocity: float = 3.0
    turn_velocity: float = 2.0

    def __post_init__(self) -> None:
        if self.forward_velocity <= 0.0:
            raise ValueError("forward_velocity must be positive.")
        if self.turn_velocity <= 0.0:
            raise ValueError("turn_velocity must be positive.")

    def command_for(
        self,
        primitive: MotionPrimitive | int,
    ) -> WheelCommand:
        try:
            resolved = MotionPrimitive(int(primitive))
        except (TypeError, ValueError) as error:
            raise ValueError(f"Unsupported motion primitive: {primitive!r}") from error

        if resolved in {
            MotionPrimitive.STOP,
            MotionPrimitive.INSPECT,
        }:
            return WheelCommand(0.0, 0.0)
        if resolved is MotionPrimitive.MOVE_FORWARD:
            return WheelCommand(
                self.forward_velocity,
                self.forward_velocity,
            )
        if resolved is MotionPrimitive.TURN_LEFT:
            return WheelCommand(
                -self.turn_velocity,
                self.turn_velocity,
            )
        if resolved is MotionPrimitive.TURN_RIGHT:
            return WheelCommand(
                self.turn_velocity,
                -self.turn_velocity,
            )

        raise AssertionError(f"Unhandled motion primitive: {resolved}")
