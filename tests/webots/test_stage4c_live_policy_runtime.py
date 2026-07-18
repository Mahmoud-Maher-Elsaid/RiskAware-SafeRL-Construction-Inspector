from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]

VALIDATOR_PATH = PROJECT_ROOT / "scripts" / "validate_stage4c_live_policy_runtime.py"

ROBOT_CONTROLLER_PATH = (
    PROJECT_ROOT / "webots" / "controllers" / "live_bridge_robot" / "live_bridge_robot.py"
)

RUNNER_PATH = PROJECT_ROOT / "scripts" / "run_stage4c_live_policy_dry_run.ps1"


def load_validator_module():
    specification = importlib.util.spec_from_file_location(
        "stage4c_runtime_validator",
        VALIDATOR_PATH,
    )

    assert specification is not None
    assert specification.loader is not None

    module = importlib.util.module_from_spec(specification)

    specification.loader.exec_module(module)

    return module


def valid_policy_record() -> dict[str, object]:
    return {
        "schema_version": 1,
        "sample_index": 0,
        "action": 0,
        "action_name": "MOVE_UP",
        "deterministic": True,
        "action_mask": [
            True,
            False,
            True,
            False,
            False,
        ],
        "valid_action_count": 2,
        "mask_respected": True,
        "motors_connected": False,
        "simulation_time": 0.512,
        "world_x": 0.0,
        "world_z": 0.0,
        "grid_row": 5,
        "grid_column": 5,
        "heading": "EAST",
        "observation_source": ("live_webots_telemetry_reconstruction"),
        "live_map_verified": True,
        "live_state_verified": True,
        "live_action_mask_verified": True,
        "checkpoint_sha256": ("172437CAE45B69031F443C0707FB0795D2F1860D3B95594BE281645D8A173FE7"),
        "proposal_only": True,
        "policy_action_applied": False,
        "policy_controls_motors": False,
        "motor_command_channel": "none",
        "robot_controller_modified": False,
    }


def test_valid_policy_runtime_record_passes() -> None:
    validator = load_validator_module()

    validator.validate_policy_record(valid_policy_record())


def test_masked_policy_action_is_rejected() -> None:
    validator = load_validator_module()

    record = valid_policy_record()
    record["action"] = 1
    record["action_name"] = "MOVE_DOWN"

    with pytest.raises(
        RuntimeError,
        match="masked-out",
    ):
        validator.validate_policy_record(record)


def test_motor_connection_is_rejected() -> None:
    validator = load_validator_module()

    record = valid_policy_record()
    record["policy_controls_motors"] = True

    with pytest.raises(
        RuntimeError,
        match="controls the motors",
    ):
        validator.validate_policy_record(record)


def test_robot_controller_remains_policy_free() -> None:
    content = ROBOT_CONTROLLER_PATH.read_text(encoding="utf-8")

    assert "PolicyDryRunEngine" not in content
    assert "policy_dry_run" not in content

    assert "resolve_wheel_velocities(simulation_time)" in content

    assert "left_motor.setVelocity(left_velocity)" in content

    assert "right_motor.setVelocity(right_velocity)" in content


def test_stage4c_runner_uses_sidecar_isolation() -> None:
    content = RUNNER_PATH.read_text(encoding="utf-8")

    assert "stage4c_live_policy_sidecar.py" in content

    assert "run_stage4b2_runtime.ps1" in content

    assert "stage4c_live_policy_runtime_summary.json" in content


def test_stage4c_runner_uses_artifact_based_process_validation() -> None:
    content = RUNNER_PATH.read_text(encoding="utf-8")

    assert "$SidecarProcess.WaitForExit()" in content
    assert "$SidecarProcess.HasExited" in content

    assert "STAGE4C_SIDECAR_COMPLETE proposals=30" in content

    assert "The policy sidecar created a traceback file." in content

    assert "$SidecarProcess.ExitCode" not in content
