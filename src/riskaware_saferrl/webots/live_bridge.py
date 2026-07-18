from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import Any

import numpy as np

from riskaware_saferrl.webots.bridge import (
    CardinalHeading,
    GridFrame,
    WebotsSensorSnapshot,
)

LIVE_BRIDGE_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class LiveBridgeTelemetry:
    schema_version: int
    sample_index: int
    simulation_time: float
    world_x: float
    world_z: float
    yaw_radians: float
    grid_row: int
    grid_column: int
    heading: str
    proximity: tuple[float, ...]
    compass: tuple[float, float, float]
    map_length: int
    map_nonzero: int
    map_sum: float
    state: tuple[float, float, float, float]
    action_mask: tuple[bool, bool, bool, bool, bool]
    ack_count: int = 0

    def __post_init__(self) -> None:
        if self.schema_version != LIVE_BRIDGE_SCHEMA_VERSION:
            raise ValueError("Unsupported live bridge schema version.")

        if self.sample_index < 0:
            raise ValueError("sample_index must be non-negative.")

        if self.simulation_time < 0.0:
            raise ValueError("simulation_time must be non-negative.")

        finite_values = (
            self.simulation_time,
            self.world_x,
            self.world_z,
            self.yaw_radians,
            self.map_sum,
            *self.proximity,
            *self.compass,
            *self.state,
        )

        if not all(math.isfinite(value) for value in finite_values):
            raise ValueError("Telemetry numeric values must be finite.")

        if not (0 <= self.grid_row < 12 and 0 <= self.grid_column < 12):
            raise ValueError("Telemetry grid position is outside the 12 by 12 grid.")

        valid_headings = {heading.name for heading in CardinalHeading}

        if self.heading not in valid_headings:
            raise ValueError(f"Unsupported cardinal heading: {self.heading}")

        if len(self.proximity) != 8:
            raise ValueError("Exactly eight proximity readings are required.")

        if len(self.compass) != 3:
            raise ValueError("Exactly three compass values are required.")

        if self.map_length != 7 * 12 * 12:
            raise ValueError("map_length must equal 1008.")

        if not (0 <= self.map_nonzero <= self.map_length):
            raise ValueError("map_nonzero is outside the valid range.")

        if len(self.state) != 4:
            raise ValueError("The state vector must contain four values.")

        if len(self.action_mask) != 5:
            raise ValueError("The action mask must contain five values.")

        if not any(self.action_mask):
            raise ValueError("The action mask must contain a valid action.")

        if self.ack_count < 0:
            raise ValueError("ack_count must be non-negative.")

    @classmethod
    def from_observation(
        cls,
        *,
        sample_index: int,
        simulation_time: float,
        snapshot: WebotsSensorSnapshot,
        frame: GridFrame,
        compass: tuple[float, float, float],
        observation: dict[str, np.ndarray],
        action_mask: np.ndarray,
        ack_count: int = 0,
    ) -> LiveBridgeTelemetry:
        if set(observation) != {
            "map",
            "state",
        }:
            raise ValueError("Observation must contain map and state.")

        map_values = np.asarray(
            observation["map"],
            dtype=np.float32,
        ).reshape(-1)

        state_values = np.asarray(
            observation["state"],
            dtype=np.float32,
        ).reshape(-1)

        mask_values = np.asarray(
            action_mask,
            dtype=np.bool_,
        ).reshape(-1)

        if map_values.size != 7 * 12 * 12:
            raise ValueError("The flattened semantic map must contain 1008 values.")

        if state_values.size != 4:
            raise ValueError("The observation state must contain four values.")

        if mask_values.size != 5:
            raise ValueError("The action mask must contain five values.")

        grid_row, grid_column = snapshot.grid_position(frame)

        return cls(
            schema_version=LIVE_BRIDGE_SCHEMA_VERSION,
            sample_index=sample_index,
            simulation_time=float(simulation_time),
            world_x=float(snapshot.x),
            world_z=float(snapshot.z),
            yaw_radians=float(snapshot.yaw_radians),
            grid_row=grid_row,
            grid_column=grid_column,
            heading=snapshot.heading.name,
            proximity=tuple(float(value) for value in snapshot.proximity),
            compass=tuple(float(value) for value in compass),
            map_length=int(map_values.size),
            map_nonzero=int(np.count_nonzero(map_values)),
            map_sum=float(
                np.sum(
                    map_values,
                    dtype=np.float64,
                )
            ),
            state=tuple(float(value) for value in state_values),
            action_mask=tuple(bool(value) for value in mask_values),
            ack_count=ack_count,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "sample_index": self.sample_index,
            "simulation_time": self.simulation_time,
            "world_x": self.world_x,
            "world_z": self.world_z,
            "yaw_radians": self.yaw_radians,
            "grid_row": self.grid_row,
            "grid_column": self.grid_column,
            "heading": self.heading,
            "proximity": list(self.proximity),
            "compass": list(self.compass),
            "map_length": self.map_length,
            "map_nonzero": self.map_nonzero,
            "map_sum": self.map_sum,
            "state": list(self.state),
            "action_mask": list(self.action_mask),
            "ack_count": self.ack_count,
        }

    def to_json(self) -> str:
        return json.dumps(
            self.to_dict(),
            separators=(",", ":"),
            sort_keys=True,
        )

    @classmethod
    def from_json(
        cls,
        encoded: str,
    ) -> LiveBridgeTelemetry:
        try:
            payload = json.loads(encoded)
        except json.JSONDecodeError as error:
            raise ValueError("Invalid live bridge telemetry payload.") from error

        try:
            schema_version = int(payload["schema_version"])
            sample_index = int(payload["sample_index"])
            simulation_time = float(payload["simulation_time"])
            world_x = float(payload["world_x"])
            world_z = float(payload["world_z"])
            yaw_radians = float(payload["yaw_radians"])
            grid_row = int(payload["grid_row"])
            grid_column = int(payload["grid_column"])
            heading = str(payload["heading"])
            proximity = tuple(float(value) for value in payload["proximity"])
            compass = tuple(float(value) for value in payload["compass"])
            map_length = int(payload["map_length"])
            map_nonzero = int(payload["map_nonzero"])
            map_sum = float(payload["map_sum"])
            state = tuple(float(value) for value in payload["state"])
            action_mask = tuple(bool(value) for value in payload["action_mask"])
            ack_count = int(
                payload.get(
                    "ack_count",
                    0,
                )
            )
        except (
            KeyError,
            TypeError,
            ValueError,
        ) as error:
            raise ValueError("Invalid live bridge telemetry payload.") from error

        return cls(
            schema_version=schema_version,
            sample_index=sample_index,
            simulation_time=simulation_time,
            world_x=world_x,
            world_z=world_z,
            yaw_radians=yaw_radians,
            grid_row=grid_row,
            grid_column=grid_column,
            heading=heading,
            proximity=proximity,
            compass=compass,
            map_length=map_length,
            map_nonzero=map_nonzero,
            map_sum=map_sum,
            state=state,
            action_mask=action_mask,
            ack_count=ack_count,
        )
