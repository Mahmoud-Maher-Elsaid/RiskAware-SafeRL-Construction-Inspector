from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def test_stage4a_world_is_self_contained() -> None:
    world_path = PROJECT_ROOT / "webots" / "worlds" / "construction_site_stage4a.wbt"
    content = world_path.read_text(encoding="utf-8")

    assert "EXTERNPROTO" not in content
    assert "TexturedBackground" not in content
    assert "DEF CONSTRUCTION_ROBOT Robot" in content
    assert 'controller "manual_robot_test"' in content
    assert 'controller "scenario_supervisor"' in content
    assert 'name "left wheel motor"' in content
    assert 'name "right wheel motor"' in content
    assert 'name "left wheel sensor"' in content
    assert 'name "right wheel sensor"' in content

    for sensor_index in range(8):
        assert f'name "ps{sensor_index}"' in content


def test_stage4a_controller_entrypoints_exist() -> None:
    required_files = [
        PROJECT_ROOT / "webots" / "controllers" / "manual_robot_test" / "manual_robot_test.py",
        PROJECT_ROOT / "webots" / "controllers" / "scenario_supervisor" / "scenario_supervisor.py",
        PROJECT_ROOT / "webots" / "controllers" / "construction_robot" / "construction_robot.py",
        PROJECT_ROOT / "scripts" / "validate_stage4a_runtime.py",
    ]

    for path in required_files:
        assert path.is_file(), path
        assert path.stat().st_size > 0, path
