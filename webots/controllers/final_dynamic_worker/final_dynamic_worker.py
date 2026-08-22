from __future__ import annotations

import json
import math
import os
import time
from pathlib import Path

from controller import Supervisor


def main() -> None:
    output = Path(os.environ.get("RISK_AWARE_EXPERIMENTAL_OUTPUT", "."))
    output.mkdir(parents=True, exist_ok=True)
    (output / "marker_worker_main_started.json").write_text(
        json.dumps({"timestamp": time.time()}) + "\n", encoding="utf-8"
    )
    supervisor = Supervisor()
    timestep = int(supervisor.getBasicTimeStep())
    node = supervisor.getSelf()
    translation = node.getField("translation")
    origin = translation.getSFVec3f()
    scenario_seed = int(os.environ.get("RISK_AWARE_SCENARIO_SEED", "42"))
    phase = (scenario_seed % 360) * math.pi / 180.0
    offset = ((scenario_seed * 37) % 100 - 50) / 200.0
    elapsed = 0.0
    while supervisor.step(timestep) != -1:
        elapsed += timestep / 1000.0
        translation.setSFVec3f(
            [
                origin[0] + offset + 1.5 * math.sin(0.35 * elapsed + phase),
                origin[1],
                origin[2],
            ]
        )


if __name__ == "__main__":
    main()
