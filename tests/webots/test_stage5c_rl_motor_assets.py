from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def test_stage5c_controller_connects_policy_trace_to_motors() -> None:
    path = (
        PROJECT_ROOT
        / "webots"
        / "controllers"
        / "rl_autonomous_inspection_robot"
        / "rl_autonomous_inspection_robot.py"
    )
    content = path.read_text(encoding="utf-8")
    required = (
        "PolicyDryRunEngine.from_checkpoint(",
        "SafePolicyPipeline(",
        "policy_controls_motors=True",
        "trace.motor_commands",
        "left_motor.setVelocity(actual_command[0])",
        "right_motor.setVelocity(actual_command[1])",
        '"fallback_controller_used": False',
        "inject_perception_risk(",
    )
    for token in required:
        assert token in content


def test_stage5c_world_builder_preserves_human_view_and_dedicated_controller() -> None:
    content = (PROJECT_ROOT / "scripts" / "build_stage5c_rl_motor_world.py").read_text(
        encoding="utf-8"
    )
    assert "DEF HUMAN_VIEWPOINT Viewpoint {" in content
    assert 'controller "rl_autonomous_inspection_robot"' in content
    assert 'controller "rl_autonomous_inspection_supervisor"' in content
    assert "construction_site_stage5c_rl_motor_runtime.wbt" in content


def test_stage5c_launcher_enforces_truth_flags() -> None:
    content = (PROJECT_ROOT / "scripts" / "run_stage5c_rl_motor_runtime.ps1").read_text(
        encoding="utf-8"
    )
    for token in (
        '"policy_controls_motors"',
        '"safety_shield_active"',
        '"perception_affects_runtime_state"',
        "fallback_controller_used",
        "motor_command_change_count",
        "safety_shield_interventions",
        "first_person_final.png",
        "$env:Path",
    ):
        assert token in content
