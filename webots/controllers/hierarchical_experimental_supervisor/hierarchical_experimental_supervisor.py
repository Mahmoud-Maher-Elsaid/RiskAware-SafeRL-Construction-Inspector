from __future__ import annotations

import os
from pathlib import Path

from controller import Supervisor


def main() -> int:
    out = Path(os.environ["RISK_AWARE_EXPERIMENTAL_OUTPUT"])
    out.mkdir(parents=True, exist_ok=True)
    supervisor = Supervisor()
    timestep = int(supervisor.getBasicTimeStep())
    if supervisor.getFromDef("SHOWCASE_ROBOT") is None:
        raise RuntimeError("Experimental world is missing SHOWCASE_ROBOT")
    initial = False
    for _ in range(10000):
        if supervisor.step(timestep) == -1:
            break
        if not initial and supervisor.getTime() >= 1.0:
            supervisor.exportImage(str(out / "first_person_initial.png"), 80)
            initial = True
        if (out / "complete.marker").is_file():
            supervisor.exportImage(str(out / "first_person_final.png"), 80)
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
