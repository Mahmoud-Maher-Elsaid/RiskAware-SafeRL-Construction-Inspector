from __future__ import annotations

import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONTROLLER_PATH = (
    PROJECT_ROOT
    / "webots"
    / "controllers"
    / "closed_loop_inspection_robot"
    / "closed_loop_inspection_robot.py"
)
ROUTE_PATH = PROJECT_ROOT / "configs" / "webots" / "stage5a3_closed_loop_route.json"


def test_controller_turns_before_driving() -> None:
    content = CONTROLLER_PATH.read_text(encoding="utf-8")

    assert "ALIGNMENT_HEADING_LIMIT_RADIANS = math.radians(5.0)" in content
    assert "if absolute_heading_error > ALIGNMENT_HEADING_LIMIT_RADIANS:" in content
    assert "forward_command = 0.0" in content


def test_controller_uses_conservative_limits() -> None:
    content = CONTROLLER_PATH.read_text(encoding="utf-8")

    required_tokens = (
        "cruise_velocity = 0.20 * maximum_motor_velocity",
        "minimum_forward_velocity = 0.06 * maximum_motor_velocity",
        "turn_velocity_limit = 0.025 * maximum_motor_velocity",
        "calibration_forward_velocity = 0.10 * maximum_motor_velocity",
        "calibration_turn_velocity = 0.025 * maximum_motor_velocity",
    )

    for token in required_tokens:
        assert token in content


def test_controller_keeps_meaningful_calibration_and_tilt_guards() -> None:
    content = CONTROLLER_PATH.read_text(encoding="utf-8")

    assert "CALIBRATION_MINIMUM_YAW_CHANGE = math.radians(8.0)" in content
    assert "MAXIMUM_SAFE_ROLL_RADIANS = math.radians(10.0)" in content
    assert "MAXIMUM_SAFE_PITCH_RADIANS = math.radians(10.0)" in content
    assert "MAXIMUM_HARD_TILT_RADIANS = math.radians(18.0)" in content
    assert 'STATE_CALIBRATE_FORWARD_SETTLE = "CALIBRATE_FORWARD_SETTLE"' in content
    assert 'STATE_CALIBRATE_TURN_SETTLE = "CALIBRATE_TURN_SETTLE"' in content
    assert "if abs(yaw_change) >= CALIBRATION_MINIMUM_YAW_CHANGE:" in content


def test_route_has_stable_runtime_budget() -> None:
    route = json.loads(ROUTE_PATH.read_text(encoding="utf-8"))

    assert route["maximum_simulation_time_seconds"] == 480.0
