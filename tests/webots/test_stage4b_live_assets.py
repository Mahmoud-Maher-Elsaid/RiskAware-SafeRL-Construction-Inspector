from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

SOURCE_WORLD = PROJECT_ROOT / "webots" / "worlds" / "construction_site_stage4a.wbt"

LIVE_WORLD = PROJECT_ROOT / "webots" / "worlds" / "construction_site_stage4b_live_bridge.wbt"

ROBOT_CONTROLLER = (
    PROJECT_ROOT / "webots" / "controllers" / "live_bridge_robot" / "live_bridge_robot.py"
)

SUPERVISOR_CONTROLLER = (
    PROJECT_ROOT / "webots" / "controllers" / "live_bridge_supervisor" / "live_bridge_supervisor.py"
)


def test_live_world_exists() -> None:
    assert LIVE_WORLD.is_file()


def test_live_controllers_exist() -> None:
    assert ROBOT_CONTROLLER.is_file()
    assert SUPERVISOR_CONTROLLER.is_file()


def test_live_world_uses_new_controllers() -> None:
    content = LIVE_WORLD.read_text(encoding="utf-8")

    assert 'controller "live_bridge_robot"' in content

    assert 'controller "live_bridge_supervisor"' in content

    assert 'controller "scenario_supervisor"' not in content


def test_live_world_contains_pose_sensors() -> None:
    content = LIVE_WORLD.read_text(encoding="utf-8")

    assert 'name "gps"' in content
    assert 'name "inertial unit"' in content
    assert 'name "compass"' in content


def test_live_world_contains_communication_devices() -> None:
    content = LIVE_WORLD.read_text(encoding="utf-8")

    assert content.count('name "bridge emitter"') == 2

    assert content.count('name "bridge receiver"') == 2

    assert "channel 7" in content
    assert "channel 8" in content


def test_validated_stage4a_world_is_unchanged() -> None:
    content = SOURCE_WORLD.read_text(encoding="utf-8")

    assert 'controller "manual_robot_test"' in content

    assert 'controller "scenario_supervisor"' in content

    assert 'controller "live_bridge_robot"' not in content


def test_live_world_uses_eun_coordinates() -> None:
    content = LIVE_WORLD.read_text(encoding="utf-8")

    assert content.count('coordinateSystem "EUN"') == 1
