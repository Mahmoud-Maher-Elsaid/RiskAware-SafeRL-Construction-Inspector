from __future__ import annotations

import math

from controller import Supervisor


def main() -> None:
    supervisor = Supervisor()
    timestep = int(supervisor.getBasicTimeStep())
    node = supervisor.getSelf()
    translation = node.getField("translation")
    origin = translation.getSFVec3f()
    elapsed = 0.0
    while supervisor.step(timestep) != -1:
        elapsed += timestep / 1000.0
        translation.setSFVec3f([origin[0] + 1.5 * math.sin(0.35 * elapsed), origin[1], origin[2]])


if __name__ == "__main__":
    main()
