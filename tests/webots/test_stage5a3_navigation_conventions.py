from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

CONTROLLER_PATH = (
    PROJECT_ROOT
    / "webots"
    / "controllers"
    / "closed_loop_inspection_robot"
    / "closed_loop_inspection_robot.py"
)


def controller_content() -> str:
    return CONTROLLER_PATH.read_text(encoding="utf-8")


def test_camera_forward_motor_convention() -> None:
    import re

    content = controller_content()

    assert "FORWARD_MOTOR_SIGN = -1.0" in content

    assert re.search(
        r"left_command\s*=\s*(?:\(\s*)?"
        r"FORWARD_MOTOR_SIGN\s*\*\s*"
        r"calibration_forward_velocity",
        content,
    )

    assert re.search(
        r"forward_command\s*=\s*(?:\(\s*)?"
        r"FORWARD_MOTOR_SIGN\s*\*\s*"
        r"distance_velocity\s*\*\s*alignment_scale",
        content,
    )


def test_eun_yaw_mapping_is_explicit() -> None:
    content = controller_content()

    assert "YAW_TO_WORLD_SIGN = -1.0" in content
    assert "def y_up_attitude_from_quaternion(" in content
    assert "inertial_unit.getQuaternion()" in content
    assert "math.atan2(float(compass_values[0]), float(compass_values[2]))" in content

    assert "YAW_TO_WORLD_SIGN * current_yaw + heading_offset" in content

    assert "- YAW_TO_WORLD_SIGN * float(calibration_start_yaw)" in content


def test_turn_calibration_maps_to_world_heading() -> None:
    content = controller_content()

    assert "turn_command_sign = YAW_TO_WORLD_SIGN * (" in content


def test_stagnation_watchdog_ignores_alignment_turning() -> None:
    content = controller_content()

    assert "heading_aligned_for_progress" in content

    assert "if not heading_aligned_for_progress:" in content

    assert "heading_aligned_for_progress\n                    and simulation_time" in content


def test_repaired_controller_uses_smooth_gains() -> None:
    content = controller_content()

    assert "NAVIGATION_HEADING_GAIN = 0.45" in content
    assert "NAVIGATION_DISTANCE_GAIN = 0.75" in content

    assert "ALIGNMENT_HEADING_LIMIT_RADIANS = math.radians(5.0)" in content

    assert "SCAN_HEADING_GAIN = 0.55" in content
    assert "STAGNATION_TIMEOUT_SECONDS = 12.0" in content


def test_mission_summary_records_navigation_conventions() -> None:
    content = controller_content()

    assert '"yaw_to_world_sign": (YAW_TO_WORLD_SIGN)' in content

    assert '"forward_motor_sign": (FORWARD_MOTOR_SIGN)' in content
