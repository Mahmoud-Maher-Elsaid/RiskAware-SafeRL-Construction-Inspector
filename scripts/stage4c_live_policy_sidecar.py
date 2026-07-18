from __future__ import annotations

import json
import math
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np

from riskaware_saferrl.webots import (
    BridgeState,
    ObservationBridge,
    SemanticScene,
)
from riskaware_saferrl.webots.policy_dry_run import (
    PolicyDryRunEngine,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]

ROBOT_LOG_PATH = PROJECT_ROOT / "webots" / "logs" / "stage4b2_live_robot.jsonl"

PROPOSAL_LOG_PATH = PROJECT_ROOT / "webots" / "logs" / "stage4c_live_policy_proposals.jsonl"

STARTUP_PATH = PROJECT_ROOT / "webots" / "logs" / "stage4c_policy_sidecar_startup.txt"

ERROR_PATH = PROJECT_ROOT / "webots" / "logs" / "stage4c_policy_sidecar_error.log"

COMPLETION_PATH = PROJECT_ROOT / "webots" / "logs" / "stage4c_policy_sidecar_complete.txt"

CHECKPOINT_PATH = (
    PROJECT_ROOT
    / "artifacts"
    / "runs"
    / "maskable_ppo_deadlock_safe_shield_seed42_u100"
    / "evaluations"
    / "best_model"
    / "best_model.zip"
)

EXPECTED_SHA256 = "172437CAE45B69031F443C0707FB0795D2F1860D3B95594BE281645D8A173FE7"

TARGET_SAMPLE_COUNT = 30
MAXIMUM_WALL_TIME_SECONDS = 140.0
POLL_INTERVAL_SECONDS = 0.05


def require(
    condition: bool,
    message: str,
) -> None:
    if not condition:
        raise RuntimeError(message)


def load_jsonl_if_available(
    path: Path,
) -> list[dict[str, Any]]:
    if not path.is_file():
        return []

    records: list[dict[str, Any]] = []

    for line_number, raw_line in enumerate(
        path.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        line = raw_line.strip()

        if not line:
            continue

        try:
            payload = json.loads(line)
        except json.JSONDecodeError as error:
            raise RuntimeError(f"Invalid JSON on line {line_number} of {path.name}.") from error

        if not isinstance(
            payload,
            dict,
        ):
            raise RuntimeError(f"Expected an object on line {line_number} of {path.name}.")

        records.append(payload)

    return records


def append_jsonl(
    path: Path,
    payload: dict[str, object],
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open(
        "a",
        encoding="utf-8",
        newline="\n",
    ) as output_file:
        output_file.write(
            json.dumps(
                payload,
                sort_keys=True,
            )
            + "\n"
        )


def validate_live_record(
    record: dict[str, Any],
) -> None:
    require(
        int(record["schema_version"]) == 1,
        "Unexpected live telemetry schema version.",
    )

    require(
        int(record["map_length"]) == 1008,
        "Live telemetry map length must equal 1008.",
    )

    require(
        len(list(record["state"])) == 4,
        "Live telemetry state must contain four values.",
    )

    require(
        len(list(record["action_mask"])) == 5,
        "Live telemetry action mask must contain five values.",
    )

    require(
        any(bool(value) for value in record["action_mask"]),
        "Live telemetry contains an empty action mask.",
    )

    numeric_values = [
        record["simulation_time"],
        record["world_x"],
        record["world_z"],
        record["yaw_radians"],
        record["map_sum"],
        *record["state"],
    ]

    require(
        all(math.isfinite(float(value)) for value in numeric_values),
        "Live telemetry contains a non-finite value.",
    )


def create_scene() -> SemanticScene:
    return SemanticScene(
        size=12,
        obstacles=frozenset(
            {
                (5, 6),
                (7, 5),
            }
        ),
        hazards=frozenset(
            {
                (6, 8),
                (4, 7),
            }
        ),
        workers=frozenset(
            {
                (7, 7),
            }
        ),
        restricted=frozenset(
            {
                (6, 4),
            }
        ),
    )


def build_verified_observation(
    *,
    record: dict[str, Any],
    scene: SemanticScene,
    observation_bridge: ObservationBridge,
    visited: set[tuple[int, int]],
) -> tuple[
    dict[str, np.ndarray],
    np.ndarray,
    tuple[int, int],
]:
    validate_live_record(record)

    sample_index = int(record["sample_index"])

    position = (
        int(record["grid_row"]),
        int(record["grid_column"]),
    )

    require(
        0 <= position[0] < 12,
        "Live grid row is invalid.",
    )

    require(
        0 <= position[1] < 12,
        "Live grid column is invalid.",
    )

    require(
        position not in scene.obstacles,
        "Live robot position overlaps a semantic obstacle.",
    )

    visited.add(position)

    bridge_state = BridgeState(
        agent_position=position,
        visited=frozenset(visited),
        inspected=frozenset(),
        steps=sample_index,
        max_steps=250,
    )

    observation = observation_bridge.build_observation(
        scene,
        bridge_state,
    )

    action_mask = observation_bridge.action_mask(
        scene,
        bridge_state,
    )

    map_values = np.asarray(
        observation["map"],
        dtype=np.float32,
    ).reshape(-1)

    state_values = np.asarray(
        observation["state"],
        dtype=np.float32,
    ).reshape(-1)

    recorded_state = np.asarray(
        record["state"],
        dtype=np.float32,
    ).reshape(-1)

    recorded_mask = np.asarray(
        record["action_mask"],
        dtype=np.bool_,
    ).reshape(-1)

    require(
        map_values.size == 1008,
        "Reconstructed map length is invalid.",
    )

    require(
        state_values.size == 4,
        "Reconstructed state length is invalid.",
    )

    require(
        np.array_equal(
            action_mask,
            recorded_mask,
        ),
        "Reconstructed action mask does not match live telemetry.",
    )

    require(
        bool(
            np.allclose(
                state_values,
                recorded_state,
                rtol=0.0,
                atol=1e-6,
            )
        ),
        "Reconstructed state does not match live telemetry.",
    )

    require(
        int(np.count_nonzero(map_values)) == int(record["map_nonzero"]),
        "Reconstructed map_nonzero does not match live telemetry.",
    )

    require(
        math.isclose(
            float(
                np.sum(
                    map_values,
                    dtype=np.float64,
                )
            ),
            float(record["map_sum"]),
            rel_tol=0.0,
            abs_tol=1e-6,
        ),
        "Reconstructed map sum does not match live telemetry.",
    )

    return (
        observation,
        action_mask,
        position,
    )


def main() -> None:
    PROPOSAL_LOG_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    STARTUP_PATH.write_text(
        "policy sidecar entered main\n",
        encoding="utf-8",
    )

    engine = PolicyDryRunEngine.from_checkpoint(
        CHECKPOINT_PATH,
        expected_sha256=EXPECTED_SHA256,
        device="cpu",
        seed=42,
    )

    scene = create_scene()

    observation_bridge = ObservationBridge(
        vision_radius=4,
        inspection_radius=2,
    )

    visited: set[tuple[int, int]] = set()

    processed_count = 0
    previous_sample_index = -1

    deadline = time.monotonic() + MAXIMUM_WALL_TIME_SECONDS

    print(
        "STAGE4C_SIDECAR_READY",
        flush=True,
    )

    while processed_count < TARGET_SAMPLE_COUNT:
        if time.monotonic() >= deadline:
            raise RuntimeError("Stage 4C sidecar timed out before receiving 30 samples.")

        robot_records = load_jsonl_if_available(ROBOT_LOG_PATH)

        while processed_count < len(robot_records) and processed_count < TARGET_SAMPLE_COUNT:
            record = robot_records[processed_count]

            sample_index = int(record["sample_index"])

            require(
                sample_index > previous_sample_index,
                "Live telemetry sample indices are not strictly increasing.",
            )

            (
                observation,
                action_mask,
                position,
            ) = build_verified_observation(
                record=record,
                scene=scene,
                observation_bridge=observation_bridge,
                visited=visited,
            )

            proposal = engine.propose(
                sample_index=sample_index,
                observation=observation,
                action_mask=action_mask,
                deterministic=True,
            )

            proposal_payload = proposal.to_dict()

            proposal_payload.update(
                {
                    "simulation_time": float(record["simulation_time"]),
                    "world_x": float(record["world_x"]),
                    "world_z": float(record["world_z"]),
                    "grid_row": position[0],
                    "grid_column": position[1],
                    "heading": str(record["heading"]),
                    "observation_source": ("live_webots_telemetry_reconstruction"),
                    "live_map_verified": True,
                    "live_state_verified": True,
                    "live_action_mask_verified": True,
                    "checkpoint_sha256": (engine.checkpoint_hash),
                    "proposal_only": True,
                    "policy_action_applied": False,
                    "policy_controls_motors": False,
                    "motor_command_channel": "none",
                    "robot_controller_modified": False,
                }
            )

            append_jsonl(
                PROPOSAL_LOG_PATH,
                proposal_payload,
            )

            print(
                "STAGE4C_LIVE_PROPOSAL "
                f"sample={sample_index} "
                f"action={proposal.action_name} "
                f"grid={position}",
                flush=True,
            )

            previous_sample_index = sample_index

            processed_count += 1

        if processed_count < TARGET_SAMPLE_COUNT:
            time.sleep(POLL_INTERVAL_SECONDS)

    COMPLETION_PATH.write_text(
        (f"policy sidecar completed {processed_count} proposals\n"),
        encoding="utf-8",
    )

    print(
        f"STAGE4C_SIDECAR_COMPLETE proposals={processed_count}",
        flush=True,
    )


if __name__ == "__main__":
    try:
        main()
    except Exception:
        ERROR_PATH.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        ERROR_PATH.write_text(
            traceback.format_exc(),
            encoding="utf-8",
        )

        raise
