from __future__ import annotations

import os
import time
from pathlib import Path

from controller import Supervisor


def main() -> int:
    out = Path(os.environ["RISK_AWARE_EXPERIMENTAL_OUTPUT"])
    out.mkdir(parents=True, exist_ok=True)
    supervisor = Supervisor()
    demo_duration = float(os.environ.get("RISK_AWARE_EXPERIMENTAL_DEMO_DURATION", "0"))
    demo_started = time.monotonic()
    timestep = int(supervisor.getBasicTimeStep())
    if supervisor.getFromDef("SHOWCASE_ROBOT") is None:
        raise RuntimeError("Experimental world is missing SHOWCASE_ROBOT")
    initial = False
    # Fast mode can advance thousands of simulation steps while CUDA and CV
    # initialization are still in progress. The launcher owns the wall-clock
    # bound; this loop only provides a generous simulation-step bound.
    for _ in range(2000000):
        if supervisor.step(timestep) == -1:
            break
        if not initial and supervisor.getTime() >= 1.0:
            supervisor.exportImage(str(out / "first_person_initial.png"), 80)
            initial = True
        if (out / "complete.marker").is_file():
            supervisor.exportImage(str(out / "first_person_final.png"), 80)
            if demo_duration <= 0 or time.monotonic() - demo_started >= demo_duration:
                supervisor.simulationQuit(0)
                return 0
        if (out / "failure.json").is_file():
            supervisor.simulationQuit(1)
            return 1
    (out / "failure.json").write_text(
        '{"error":"experimental runtime timeout"}\n', encoding="utf-8"
    )
    supervisor.simulationQuit(1)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
