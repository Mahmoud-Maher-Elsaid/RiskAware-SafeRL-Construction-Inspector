from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
WORLD_PATH = (
    PROJECT_ROOT / "webots" / "worlds" / "construction_site_stage5a3_closed_loop_mission.wbt"
)
SOURCE_WORLD_PATH = PROJECT_ROOT / "webots" / "worlds" / "construction_site_stage5a_live_camera.wbt"
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


def test_human_level_camera_and_mounted_viewpoint() -> None:
    for world_path in (
        SOURCE_WORLD_PATH,
        WORLD_PATH,
    ):
        content = world_path.read_text(encoding="utf-8")

        required_tokens = (
            'name "inspection camera"',
            "translation 0.35 1.55 0",
            "rotation 1 0 0 -1.5708",
            "fieldOfView 1.05",
            "width 640",
            "height 360",
            "near 0.05",
            "far 70",
            "antiAliasing TRUE",
            "orientation 0 1 0 -1.5708",
            "position -7.95 1.67 -5.4",
            ('follow "professional construction inspection robot"'),
            'followType "Mounted Shot"',
        )

        for token in required_tokens:
            assert token in content

        assert "followOrientation" not in content


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
