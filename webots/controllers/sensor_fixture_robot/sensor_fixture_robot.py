import json
import os
from pathlib import Path

from controller import Robot

robot = Robot()
sensors = [robot.getDevice(name) for name in ("probe", "probe_flip", "probe_x", "probe_x_flip")]
for sensor in sensors:
    sensor.enable(int(robot.getBasicTimeStep()))
out = Path(os.environ.get("SENSOR_FIXTURE_OUTPUT", "."))
out.mkdir(parents=True, exist_ok=True)
trace = out / "sensor_fixture_readings.jsonl"
step = 0
with trace.open("w", encoding="utf-8") as stream:
    while robot.step(int(robot.getBasicTimeStep())) != -1:
        stream.write(
            json.dumps({"step": step, "values": [float(sensor.getValue()) for sensor in sensors]})
            + "\n"
        )
        stream.flush()
        step += 1
        if step >= 125:
            break
