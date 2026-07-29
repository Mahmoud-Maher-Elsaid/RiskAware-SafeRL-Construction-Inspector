from __future__ import annotations

import math
from pathlib import Path
from types import ModuleType

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]

RUNNER_PATH = PROJECT_ROOT / "scripts" / "run_verified_human_first_person_stage5b3.ps1"

LAUNCHER_PATH = PROJECT_ROOT / "scripts" / "run_stage5a3_closed_loop_mission.py"

ROBOT_CONTROLLER_PATH = (
    PROJECT_ROOT
    / "webots"
    / "controllers"
    / "closed_loop_inspection_robot"
    / "closed_loop_inspection_robot.py"
)

SUPERVISOR_PATH = (
    PROJECT_ROOT
    / "webots"
    / "controllers"
    / "closed_loop_inspection_supervisor"
    / "closed_loop_inspection_supervisor.py"
)


def load_supervisor_helpers() -> ModuleType:
    source = SUPERVISOR_PATH.read_text(encoding="utf-8")
    helper_source = source.split("def main() -> None:", maxsplit=1)[0]
    helper_source = helper_source.replace("from controller import Supervisor\n", "")
    namespace: dict[str, object] = {
        "__file__": str(SUPERVISOR_PATH),
        "__name__": "closed_loop_inspection_supervisor_helpers",
    }
    exec(compile(helper_source, str(SUPERVISOR_PATH), "exec"), namespace)
    module = ModuleType("closed_loop_inspection_supervisor_helpers")
    for name, value in namespace.items():
        setattr(module, name, value)
    return module


def test_one_command_runner_starts_stage5b3() -> None:
    content = RUNNER_PATH.read_text(encoding="utf-8")

    assert "run_stage5b3_live_perception_mission.ps1" in content

    assert "-RepoRoot" in content
    assert "Read-Host" not in content


def test_launcher_waits_for_final_view_and_closes_cleanly() -> None:
    content = LAUNCHER_PATH.read_text(encoding="utf-8")

    required_tokens = (
        "completion_detected = False",
        "completion_detected = True",
        "time.sleep(3.5)",
        "except ValueError:",
        "output_thread.join(timeout=5.0)",
    )

    for token in required_tokens:
        assert token in content


@pytest.mark.parametrize(
    ("forward_x", "forward_z", "expected_yaw"),
    [
        (1.0, 0.0, 0.0),
        (0.0, -1.0, math.pi / 2.0),
        (-1.0, 0.0, math.pi),
        (0.0, 1.0, -math.pi / 2.0),
    ],
)
def test_viewpoint_yaw_uses_webots_positive_x_default_view_direction(
    forward_x: float,
    forward_z: float,
    expected_yaw: float,
) -> None:
    module = load_supervisor_helpers()
    actual = module.viewpoint_yaw_from_forward(forward_x, forward_z)
    delta = module.normalize_angle(actual - expected_yaw)
    assert delta == pytest.approx(0.0)


def test_zero_yaw_applies_level_human_view_rotation() -> None:
    module = load_supervisor_helpers()
    orientation = module.viewpoint_orientation_from_yaw(0.0)
    assert orientation[0] == pytest.approx(-1.0)
    assert orientation[1] == pytest.approx(0.0)
    assert orientation[2] == pytest.approx(0.0)
    assert orientation[3] == pytest.approx(math.pi / 2.0)


@pytest.mark.parametrize("yaw", [-math.pi, -math.pi / 2.0, 0.0, math.pi / 2.0, math.pi])
def test_composed_viewpoint_orientation_is_finite_and_normalized(yaw: float) -> None:
    module = load_supervisor_helpers()
    axis_x, axis_y, axis_z, angle = module.viewpoint_orientation_from_yaw(yaw)
    assert math.sqrt(axis_x**2 + axis_y**2 + axis_z**2) == pytest.approx(1.0)
    assert math.isfinite(angle)


def test_supervisor_exports_deterministic_left_and_right_turn_views() -> None:
    content = SUPERVISOR_PATH.read_text(encoding="utf-8")
    assert '"first_person_left_turn.jpg"' in content
    assert '"first_person_right_turn.jpg"' in content
    assert 'active_scan_view == "left"' in content
    assert 'active_scan_view == "right"' in content


def test_launcher_removes_stale_webots_projection_settings() -> None:
    content = LAUNCHER_PATH.read_text(encoding="utf-8")
    assert "remove_stale_world_project" in content
    assert "wbproj" in content


def test_robot_aligns_to_route_start_heading_before_final_evidence() -> None:
    content = ROBOT_CONTROLLER_PATH.read_text(encoding="utf-8")
    assert 'STATE_FINAL_ALIGN = "FINAL_ALIGN"' in content
    assert "final_heading_error = normalize_angle(-current_world_heading)" in content
    assert "state = STATE_FINAL_ALIGN" in content
