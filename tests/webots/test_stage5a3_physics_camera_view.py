from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
WORLD_PATH = (
    PROJECT_ROOT / "webots" / "worlds" / "construction_site_stage5a3_closed_loop_mission.wbt"
)
CONTROLLER_PATH = (
    PROJECT_ROOT
    / "webots"
    / "controllers"
    / "closed_loop_inspection_robot"
    / "closed_loop_inspection_robot.py"
)


def test_corrected_world_physics() -> None:
    content = WORLD_PATH.read_text(encoding="utf-8")

    required_tokens = (
        'material1 "driveWheel"',
        "coulombFriction [ 0.6 ]",
        'material1 "rearCaster"',
        "rollingFriction 0 0 0",
        'contactMaterial "driveWheel"',
        'contactMaterial "rearCaster"',
        "anchor -0.34 -0.055 0",
        "anchor 0 0 0.36",
        "anchor 0 0 -0.36",
        'name "front stabilizer ball"',
        "translation -0.34 -0.055 0",
        "-0.12 -0.055 0",
        "damping Damping {",
        "linear 0.08",
        "angular 0.85",
        "maxTorque 2.5",
    )

    for token in required_tokens:
        assert token in content

    assert content.count('contactMaterial "driveWheel"') == 2
    assert content.count("maxTorque 2.5") == 2
    assert "rollingFriction [" not in content


def test_camera_and_viewpoint_are_available() -> None:
    content = WORLD_PATH.read_text(encoding="utf-8")

    assert 'name "inspection camera"' in content
    assert "translation 0.38 0.46 0" in content
    assert "width 640" in content
    assert "height 360" in content
    assert 'follow "professional construction inspection robot"' in content
    assert "followOrientation FALSE" in content


def test_controller_has_stability_control_and_preview() -> None:
    content = CONTROLLER_PATH.read_text(encoding="utf-8")

    required_tokens = (
        "MAXIMUM_HARD_TILT_RADIANS",
        "TILT_FAILURE_DURATION_SECONDS",
        "MOTOR_COMMAND_SLEW_RATE_RADIANS_PER_SECOND",
        "def move_toward(",
        "maximum_command_delta",
        "live_camera_preview.png",
        "STAGE5A3_CAMERA_PREVIEW",
        '"dual_low_friction_passive_ball_casters"',
    )

    for token in required_tokens:
        assert token in content
