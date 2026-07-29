from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

SOURCE_WORLD_PATH = PROJECT_ROOT / "webots" / "worlds" / "construction_site_stage5a_live_camera.wbt"

MISSION_WORLD_PATH = (
    PROJECT_ROOT / "webots" / "worlds" / "construction_site_stage5a3_closed_loop_mission.wbt"
)

BUILDER_PATH = PROJECT_ROOT / "scripts" / "build_stage5a3_closed_loop_world.py"

SUPERVISOR_PATH = (
    PROJECT_ROOT
    / "webots"
    / "controllers"
    / "closed_loop_inspection_supervisor"
    / "closed_loop_inspection_supervisor.py"
)


def test_worlds_define_level_human_first_person_view() -> None:
    for world_path in (
        SOURCE_WORLD_PATH,
        MISSION_WORLD_PATH,
    ):
        content = world_path.read_text(encoding="utf-8")

        required_tokens = (
            "DEF HUMAN_VIEWPOINT Viewpoint {",
            "orientation 0 0 1 0",
            "position -7.95 1.67 -5.4",
            "fieldOfView 1.05",
            "near 0.05",
            "far 70",
            "DEF SHOWCASE_ROBOT Robot {",
            'name "inspection camera"',
            "translation 0.35 1.55 0",
            "rotation 1 0 0 -1.5708",
            "width 640",
            "height 360",
            "antiAliasing TRUE",
        )

        for token in required_tokens:
            assert token in content

        forbidden_tokens = (
            "followOrientation",
            "followSmoothness",
            'follow "professional construction inspection robot"',
            'followType "Mounted Shot"',
        )

        for token in forbidden_tokens:
            assert token not in content


def test_builder_preserves_human_view_contract() -> None:
    content = BUILDER_PATH.read_text(encoding="utf-8")

    required_tokens = (
        "DEF HUMAN_VIEWPOINT Viewpoint {",
        "orientation 0 0 1 0",
        "position -7.95 1.67 -5.4",
        "translation 0.35 1.55 0",
        "rotation 1 0 0 -1.5708",
    )

    for token in required_tokens:
        assert token in content


def test_supervisor_uses_actual_robot_orientation_matrix() -> None:
    content = SUPERVISOR_PATH.read_text(encoding="utf-8")

    required_tokens = (
        'ROBOT_DEF = "SHOWCASE_ROBOT"',
        'VIEWPOINT_DEF = "HUMAN_VIEWPOINT"',
        "robot_node.getOrientation()",
        "orientation[0]",
        "orientation[6]",
        "viewpoint_yaw_from_forward(",
        "math.atan2(",
        "-forward_z",
        "position_field.setSFVec3f(",
        "orientation_field.setSFRotation(",
        "first_person_initial.jpg",
        "first_person_middle.jpg",
        "first_person_final.jpg",
    )

    for token in required_tokens:
        assert token in content

    assert "world_heading_degrees" not in content
