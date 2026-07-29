from __future__ import annotations

from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]


def test_exact_final_world_set_and_configs_exist() -> None:
    names = ("site_small", "site_medium", "site_dynamic")
    for name in names:
        assert (ROOT / f"webots/worlds/{name}.wbt").is_file()
        assert (ROOT / f"configs/webots/{name}.yaml").is_file()


def test_world_metadata_controllers_and_first_person_view() -> None:
    for name in ("site_small", "site_medium", "site_dynamic"):
        content = (ROOT / f"webots/worlds/{name}.wbt").read_text(encoding="utf-8")
        assert f"final_world={name}" in content
        assert 'controller "rl_autonomous_inspection_robot"' in content
        assert 'controller "rl_autonomous_inspection_supervisor"' in content
        assert "DEF HUMAN_VIEWPOINT Viewpoint {" in content
        assert "fieldOfView 1.05" in content
        assert "ORTHOGRAPHIC" not in content


def test_world_configuration_metadata_is_consistent() -> None:
    configs = [
        yaml.safe_load((ROOT / f"configs/webots/{name}.yaml").read_text(encoding="utf-8"))
        for name in ("site_small", "site_medium", "site_dynamic")
    ]
    assert [config["environment_size_m"][0] for config in configs] == [24.0, 26.0, 28.0]
    assert [config["dynamic_obstacles"] for config in configs] == [False, False, True]
    assert all(1.55 <= config["first_person_eye_height_m"] <= 1.70 for config in configs)


def test_dynamic_world_has_controller_and_collision_geometry() -> None:
    content = (ROOT / "webots/worlds/site_dynamic.wbt").read_text(encoding="utf-8")
    assert 'controller "final_dynamic_worker"' in content
    assert "DEF FINAL_DYNAMIC_WORKER Robot {" in content
    assert "boundingObject Capsule" in content
