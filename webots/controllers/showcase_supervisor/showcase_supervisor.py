from __future__ import annotations

from controller import Supervisor

ROBOT_DEF = "SHOWCASE_ROBOT"

INITIAL_TRANSLATION = [
    -8.3,
    0.12,
    -5.4,
]

INITIAL_ROTATION = [
    0.0,
    1.0,
    0.0,
    0.0,
]

PHASES = (
    ("SYSTEM CHECK", 1.5),
    ("SAFE WALKWAY EAST", 8.5),
    ("TURN TO CHECKPOINT", 1.4),
    ("APPROACH INSPECTION ZONE", 6.0),
    ("SCAN LEFT", 1.4),
    ("RETURN ROUTE WEST", 8.5),
    ("TURN TO HOME LANE", 1.4),
    ("RETURN TO START", 6.0),
    ("FINAL ALIGNMENT", 1.4),
    ("SAFE STOP", 2.0),
)

CYCLE_DURATION = sum(duration for _, duration in PHASES)


def phase_name(
    simulation_time: float,
) -> str:
    cycle_time = simulation_time % CYCLE_DURATION

    elapsed = 0.0

    for name, duration in PHASES:
        elapsed += duration

        if cycle_time < elapsed:
            return name

    return PHASES[-1][0]


def set_overlay(
    supervisor: Supervisor,
    phase: str,
) -> None:
    supervisor.setLabel(
        0,
        "RiskAware SafeRL Construction Inspector",
        0.02,
        0.03,
        0.06,
        0xFFFFFF,
        0.0,
        "Arial",
    )

    supervisor.setLabel(
        1,
        "Stage 4D Professional Construction Showcase",
        0.02,
        0.095,
        0.035,
        0xFFB000,
        0.0,
        "Arial",
    )

    supervisor.setLabel(
        2,
        f"Current motion: {phase}",
        0.02,
        0.145,
        0.03,
        0x55E8FF,
        0.0,
        "Arial",
    )

    supervisor.setLabel(
        3,
        "Control mode: scripted visual demonstration",
        0.02,
        0.19,
        0.026,
        0xFFFFFF,
        0.0,
        "Arial",
    )

    supervisor.setLabel(
        4,
        "MaskablePPO motor control: DISABLED",
        0.02,
        0.23,
        0.026,
        0xFF7070,
        0.0,
        "Arial",
    )

    supervisor.setLabel(
        5,
        "Validated RL and safety pipeline remain isolated from this visual demo",
        0.02,
        0.27,
        0.023,
        0xB8FFB8,
        0.0,
        "Arial",
    )


def main() -> None:
    supervisor = Supervisor()
    time_step = int(supervisor.getBasicTimeStep())

    robot_node = supervisor.getFromDef(ROBOT_DEF)

    if robot_node is None:
        raise RuntimeError("SHOWCASE_ROBOT was not found.")

    translation_field = robot_node.getField("translation")

    rotation_field = robot_node.getField("rotation")

    previous_cycle = 0

    print(
        "STAGE4D_SHOWCASE_SUPERVISOR_READY",
        flush=True,
    )

    while supervisor.step(time_step) != -1:
        simulation_time = float(supervisor.getTime())

        current_cycle = int(simulation_time // CYCLE_DURATION)

        if current_cycle > previous_cycle:
            previous_cycle = current_cycle

            translation_field.setSFVec3f(INITIAL_TRANSLATION)

            rotation_field.setSFRotation(INITIAL_ROTATION)

            robot_node.resetPhysics()

            print(
                "STAGE4D_SHOWCASE_ROUTE_RESET",
                flush=True,
            )

        set_overlay(
            supervisor,
            phase_name(simulation_time),
        )


if __name__ == "__main__":
    main()
