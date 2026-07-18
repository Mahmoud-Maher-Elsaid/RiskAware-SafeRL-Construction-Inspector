from __future__ import annotations

import json
import traceback
from pathlib import Path

from controller import Supervisor

RECEIVER_NAME = "bridge receiver"
EMITTER_NAME = "bridge emitter"

MINIMUM_SAMPLES = 30
MAXIMUM_RUNTIME_SECONDS = 25.0

PROJECT_ROOT = Path(__file__).resolve().parents[3]

SUPERVISOR_LOG_PATH = PROJECT_ROOT / "webots" / "logs" / "stage4b2_live_supervisor.jsonl"


def append_jsonl(
    payload: dict[str, object],
) -> None:
    SUPERVISOR_LOG_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with SUPERVISOR_LOG_PATH.open(
        "a",
        encoding="utf-8",
        newline="\n",
    ) as log_file:
        log_file.write(
            json.dumps(
                payload,
                sort_keys=True,
            )
            + "\n"
        )


def main() -> None:
    supervisor = Supervisor()
    time_step = int(supervisor.getBasicTimeStep())

    receiver = supervisor.getDevice(RECEIVER_NAME)
    emitter = supervisor.getDevice(EMITTER_NAME)

    receiver.enable(time_step)

    sample_count = 0
    last_sample_index = -1

    print(
        "STAGE4B2_SUPERVISOR_READY",
        flush=True,
    )

    while supervisor.step(time_step) != -1:
        while receiver.getQueueLength() > 0:
            encoded = receiver.getString()

            try:
                payload = json.loads(encoded)

                sample_index = int(payload["sample_index"])

                if sample_index <= last_sample_index:
                    raise ValueError("Telemetry sample order is invalid.")

                last_sample_index = sample_index
                sample_count += 1

                payload["supervisor_receive_time"] = float(supervisor.getTime())

                append_jsonl(payload)

                acknowledgement = {
                    "type": "stage4b2_ack",
                    "sample_index": sample_index,
                }

                emitter.send(json.dumps(acknowledgement).encode("utf-8"))

                print(
                    f"STAGE4B2_RECEIVED sample={sample_index}",
                    flush=True,
                )
            except (
                json.JSONDecodeError,
                KeyError,
                TypeError,
                ValueError,
            ) as error:
                print(
                    f"STAGE4B2_INVALID_PACKET {error}",
                    flush=True,
                )

                supervisor.simulationQuit(2)
                return

            receiver.nextPacket()

        simulation_time = float(supervisor.getTime())

        if sample_count >= MINIMUM_SAMPLES:
            print(
                f"STAGE4B2_SUPERVISOR_COMPLETE samples={sample_count}",
                flush=True,
            )

            supervisor.simulationQuit(0)
            return

        if simulation_time >= MAXIMUM_RUNTIME_SECONDS:
            print(
                f"STAGE4B2_SUPERVISOR_TIMEOUT samples={sample_count}",
                flush=True,
            )

            supervisor.simulationQuit(2)
            return


if __name__ == "__main__":
    startup_path = PROJECT_ROOT / "webots" / "logs" / "stage4b2_supervisor_startup.txt"

    error_path = PROJECT_ROOT / "webots" / "logs" / "stage4b2_supervisor_error.log"

    startup_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    startup_path.write_text(
        "controller entered main\n",
        encoding="utf-8",
    )

    try:
        main()
    except Exception:
        error_path.write_text(
            traceback.format_exc(),
            encoding="utf-8",
        )
        raise
