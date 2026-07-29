from __future__ import annotations

from dataclasses import dataclass

from controller import Robot

LEFT_MOTOR_NAME = "left wheel motor"
RIGHT_MOTOR_NAME = "right wheel motor"


@dataclass(frozen=True)
class MotionPhase:
    name: str
    duration: float
    left_velocity: float
    right_velocity: float


PHASES = (
    MotionPhase(
        "SYSTEM CHECK",
        1.5,
        0.0,
        0.0,
    ),
    MotionPhase(
        "SAFE WALKWAY EAST",
        8.5,
        5.5,
        5.5,
    ),
    MotionPhase(
        "TURN TO CHECKPOINT",
        1.4,
        -2.5,
        2.5,
    ),
    MotionPhase(
        "APPROACH INSPECTION ZONE",
        6.0,
        5.0,
        5.0,
    ),
    MotionPhase(
        "SCAN LEFT",
        1.4,
        -2.4,
        2.4,
    ),
    MotionPhase(
        "RETURN ROUTE WEST",
        8.5,
        5.5,
        5.5,
    ),
    MotionPhase(
        "TURN TO HOME LANE",
        1.4,
        -2.5,
        2.5,
    ),
    MotionPhase(
        "RETURN TO START",
        6.0,
        5.0,
        5.0,
    ),
    MotionPhase(
        "FINAL ALIGNMENT",
        1.4,
        -2.4,
        2.4,
    ),
    MotionPhase(
        "SAFE STOP",
        2.0,
        0.0,
        0.0,
    ),
)

CYCLE_DURATION = sum(phase.duration for phase in PHASES)


def resolve_phase(
    simulation_time: float,
) -> MotionPhase:
    cycle_time = simulation_time % CYCLE_DURATION

    elapsed = 0.0

    for phase in PHASES:
        elapsed += phase.duration

        if cycle_time < elapsed:
            return phase

    return PHASES[-1]


def main() -> None:
    robot = Robot()
    time_step = int(robot.getBasicTimeStep())

    left_motor = robot.getDevice(LEFT_MOTOR_NAME)

    right_motor = robot.getDevice(RIGHT_MOTOR_NAME)

    left_motor.setPosition(float("inf"))

    right_motor.setPosition(float("inf"))

    left_motor.setVelocity(0.0)
    right_motor.setVelocity(0.0)

    previous_phase_name = ""

    print(
        "STAGE4D_SHOWCASE_ROBOT_READY",
        flush=True,
    )

    try:
        while robot.step(time_step) != -1:
            phase = resolve_phase(float(robot.getTime()))

            left_motor.setVelocity(phase.left_velocity)

            right_motor.setVelocity(phase.right_velocity)

            if phase.name != previous_phase_name:
                previous_phase_name = phase.name

                print(
                    f"STAGE4D_SHOWCASE_PHASE {phase.name}",
                    flush=True,
                )
    finally:
        left_motor.setVelocity(0.0)
        right_motor.setVelocity(0.0)

        print(
            "STAGE4D_SHOWCASE_SAFE_STOP",
            flush=True,
        )


if __name__ == "__main__":
    main()
