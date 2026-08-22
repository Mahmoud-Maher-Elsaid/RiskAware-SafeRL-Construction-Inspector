from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def test_until_closed_is_default_and_bounded_mode_remains_available() -> None:
    text = (ROOT / "scripts/run_real_v4_webots_demo.ps1").read_text(encoding="utf-8")
    assert "[string]$RunMode = 'until_closed'" in text
    assert "if($RunMode -eq 'bounded')" in text
    assert "CLOSE_WEBOTS_TO_END_THE_DEMO" in text


def test_indefinite_controller_is_webots_step_authority() -> None:
    text = (
        ROOT
        / "webots/controllers/hierarchical_experimental_robot/hierarchical_experimental_robot.py"
    ).read_text(encoding="utf-8")
    assert 'RUN_MODE = os.environ.get("RISK_AWARE_EXPERIMENTAL_RUN_MODE", "bounded")' in text
    assert 'while RUN_MODE == "until_closed" or decision_index < DECISIONS:' in text
    assert '"random_motor_noise_used": False' in text


def test_indefinite_supervisor_does_not_use_normal_timeout_quit() -> None:
    text = (
        ROOT
        / "webots/controllers/hierarchical_experimental_supervisor/hierarchical_experimental_supervisor.py"
    ).read_text(encoding="utf-8")
    assert 'run_mode = os.environ.get("RISK_AWARE_EXPERIMENTAL_RUN_MODE", "bounded")' in text
    assert 'if run_mode == "until_closed":' in text
    assert "while True:" in text


def test_visible_demo_uses_static_overview_view() -> None:
    text = (ROOT / "webots/worlds/site_dynamic_v4_visible_demo.wbt").read_text(encoding="utf-8")
    assert "DEF HUMAN_VIEWPOINT Viewpoint" in text
    assert 'follow "main reinforced concrete construction slab"' in text
    assert 'followType "Pan and Tilt Shot"' in text
    assert "Mounted Shot" not in text
    supervisor = (
        ROOT
        / "webots/controllers/hierarchical_experimental_supervisor/hierarchical_experimental_supervisor.py"
    ).read_text(encoding="utf-8")
    assert "HUMAN_VIEWPOINT" not in supervisor
    assert "FPV_ANCHOR" not in supervisor
