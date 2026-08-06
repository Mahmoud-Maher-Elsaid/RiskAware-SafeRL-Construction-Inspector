from __future__ import annotations

import json
import os
import time
from pathlib import Path

from controller import Supervisor


def main() -> int:
    out = Path(os.environ["RISK_AWARE_EXPERIMENTAL_OUTPUT"])
    out.mkdir(parents=True, exist_ok=True)
    (out / "marker_supervisor_module_imported.json").write_text(
        json.dumps({"timestamp": time.time()}) + "\n", encoding="utf-8"
    )
    supervisor = Supervisor()
    (out / "marker_supervisor_main_started.json").write_text(
        json.dumps({"timestamp": time.time()}) + "\n", encoding="utf-8"
    )
    demo_duration = float(os.environ.get("RISK_AWARE_EXPERIMENTAL_DEMO_DURATION", "0"))
    run_mode = os.environ.get("RISK_AWARE_EXPERIMENTAL_RUN_MODE", "bounded")
    demo_started = time.monotonic()
    timestep = int(supervisor.getBasicTimeStep())
    if supervisor.getFromDef("SHOWCASE_ROBOT") is None:
        raise RuntimeError("Experimental world is missing SHOWCASE_ROBOT")
    initial = False
    # Fast mode can advance thousands of simulation steps while CUDA and CV
    # initialization are still in progress. The launcher owns the wall-clock
    # bound; this loop only provides a generous simulation-step bound.
    step_count = 0
    while True:
        if supervisor.step(timestep) == -1:
            break
        step_count += 1
        if not initial:
            (out / "marker_first_supervisor_step.json").write_text(
                json.dumps({"timestamp": time.time()}) + "\n", encoding="utf-8"
            )
        if not initial and supervisor.getTime() >= 1.0:
            supervisor.exportImage(str(out / "first_person_initial.png"), 80)
            initial = True
        if run_mode == "until_closed":
            continue
        if (out / "complete.marker").is_file():
            supervisor.exportImage(str(out / "first_person_final.png"), 80)
            if demo_duration <= 0 or time.monotonic() - demo_started >= demo_duration:
                supervisor.simulationQuit(0)
                return 0
            # Realtime demonstrations must keep the mounted camera visible for
            # the requested wall-clock duration even when Webots advances faster
            # than realtime in a local process.  This is a timing hold, not a
            # route or motor controller.
            time.sleep(0.01)
        if (out / "failure.json").is_file():
            supervisor.simulationQuit(1)
            return 1
    if run_mode == "until_closed":
        return 0
    (out / "failure.json").write_text(
        '{"error":"experimental runtime timeout"}\n', encoding="utf-8"
    )
    supervisor.simulationQuit(1)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
