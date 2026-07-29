from __future__ import annotations

import math

import numpy as np
import pytest

from riskaware_saferrl.webots import (
    GridFrame,
    WebotsSensorSnapshot,
)
from riskaware_saferrl.webots.live_bridge import (
    LiveBridgeTelemetry,
)


def make_telemetry() -> LiveBridgeTelemetry:
    return LiveBridgeTelemetry(
        schema_version=1,
        sample_index=3,
        simulation_time=1.5,
        world_x=0.2,
        world_z=-0.1,
        yaw_radians=0.0,
        grid_row=6,
        grid_column=6,
        heading="EAST",
        proximity=(
            0.0,
            1.0,
            2.0,
            3.0,
            4.0,
            5.0,
            6.0,
            7.0,
        ),
        compass=(
            1.0,
            0.0,
            0.0,
        ),
        map_length=1008,
        map_nonzero=5,
        map_sum=4.5,
        state=(
            0.5,
            0.5,
            0.1,
            0.0,
        ),
        action_mask=(
            True,
            True,
            False,
            True,
            False,
        ),
        ack_count=2,
    )


def test_telemetry_json_round_trip() -> None:
    original = make_telemetry()

    restored = LiveBridgeTelemetry.from_json(original.to_json())

    assert restored == original


def test_telemetry_from_observation() -> None:
    frame = GridFrame()

    x, z = frame.grid_to_world((6, 6))

    snapshot = WebotsSensorSnapshot(
        x=x,
        z=z,
        yaw_radians=0.0,
        proximity=tuple(float(index) for index in range(8)),
    )

    semantic_map = np.zeros(
        1008,
        dtype=np.float32,
    )

    semantic_map[10] = 1.0
    semantic_map[20] = 0.5

    observation = {
        "map": semantic_map,
        "state": np.asarray(
            [
                0.5,
                0.5,
                0.2,
                0.0,
            ],
            dtype=np.float32,
        ),
    }

    mask = np.asarray(
        [
            True,
            False,
            True,
            False,
            False,
        ],
        dtype=np.bool_,
    )

    telemetry = LiveBridgeTelemetry.from_observation(
        sample_index=1,
        simulation_time=0.5,
        snapshot=snapshot,
        frame=frame,
        compass=(
            1.0,
            0.0,
            0.0,
        ),
        observation=observation,
        action_mask=mask,
    )

    assert telemetry.map_length == 1008
    assert telemetry.map_nonzero == 2
    assert telemetry.map_sum == pytest.approx(1.5)
    assert telemetry.grid_row == 6
    assert telemetry.grid_column == 6
    assert telemetry.heading == "EAST"


def test_invalid_mask_length_is_rejected() -> None:
    values = make_telemetry().to_dict()

    values["action_mask"] = [
        True,
        False,
    ]

    with pytest.raises(
        ValueError,
        match="five",
    ):
        LiveBridgeTelemetry.from_json(__import__("json").dumps(values))


def test_empty_valid_action_set_is_rejected() -> None:
    values = make_telemetry().to_dict()

    values["action_mask"] = [
        False,
        False,
        False,
        False,
        False,
    ]

    with pytest.raises(
        ValueError,
        match="valid action",
    ):
        LiveBridgeTelemetry.from_json(__import__("json").dumps(values))


def test_non_finite_value_is_rejected() -> None:
    values = make_telemetry().to_dict()

    values["world_x"] = math.inf

    with pytest.raises(
        ValueError,
        match="finite",
    ):
        LiveBridgeTelemetry.from_json(__import__("json").dumps(values))


def test_invalid_json_is_rejected() -> None:
    with pytest.raises(
        ValueError,
        match="Invalid live bridge telemetry",
    ):
        LiveBridgeTelemetry.from_json("{invalid-json")
