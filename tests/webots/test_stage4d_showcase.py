from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

WORLD_PATH = PROJECT_ROOT / "webots" / "worlds" / "construction_site_stage4d_showcase.wbt"

VALIDATION_WORLD_PATH = (
    PROJECT_ROOT / "webots" / "worlds" / "construction_site_stage4b_live_bridge.wbt"
)

ROBOT_CONTROLLER_PATH = (
    PROJECT_ROOT / "webots" / "controllers" / "showcase_robot" / "showcase_robot.py"
)

SUPERVISOR_CONTROLLER_PATH = (
    PROJECT_ROOT / "webots" / "controllers" / "showcase_supervisor" / "showcase_supervisor.py"
)

LAUNCHER_PATH = PROJECT_ROOT / "scripts" / "run_stage4d_showcase.ps1"


def test_professional_showcase_world_exists() -> None:
    assert WORLD_PATH.is_file()
    assert WORLD_PATH.stat().st_size > 20_000


def test_validation_world_is_preserved() -> None:
    assert VALIDATION_WORLD_PATH.is_file()

    content = VALIDATION_WORLD_PATH.read_text(encoding="utf-8")

    assert 'controller "live_bridge_robot"' in content


def test_showcase_world_contains_required_site_elements() -> None:
    content = WORLD_PATH.read_text(encoding="utf-8")

    required_elements = (
        "main reinforced concrete construction slab",
        "building foundation",
        "reinforced concrete column",
        "scaffold platform",
        "excavation pit bottom",
        "excavation barrier",
        "site office container",
        "tower crane mast",
        "timber pallet",
        "steel pipe",
        "inspection checkpoint",
        "professional construction inspection robot",
    )

    for required_element in required_elements:
        assert required_element in content


def test_showcase_world_uses_independent_controllers() -> None:
    content = WORLD_PATH.read_text(encoding="utf-8")

    assert 'controller "showcase_robot"' in content

    assert 'controller "showcase_supervisor"' in content


def test_showcase_robot_has_no_rl_runtime_dependency() -> None:
    content = ROBOT_CONTROLLER_PATH.read_text(encoding="utf-8")

    prohibited_tokens = (
        "gymnasium",
        "riskaware_saferrl",
        "MaskablePPO",
        "PolicyDryRunEngine",
        "stable_baselines3",
        "sb3_contrib",
    )

    for prohibited_token in prohibited_tokens:
        assert prohibited_token not in content


def test_showcase_supervisor_labels_control_scope() -> None:
    content = SUPERVISOR_CONTROLLER_PATH.read_text(encoding="utf-8")

    assert "scripted visual demonstration" in content

    assert "MaskablePPO motor control: DISABLED" in content

    assert "STAGE4D_SHOWCASE_ROUTE_RESET" in content


def test_showcase_launcher_stops_existing_webots_processes() -> None:
    content = LAUNCHER_PATH.read_text(encoding="utf-8")

    assert "Stop-AllWebotsProcesses" in content
    assert "WEBOTS_PYTHON_COMMAND" in content
    assert "--mode=realtime" in content
    assert "construction_site_stage4d_showcase.wbt" in content


def test_showcase_launcher_detects_real_webots_process() -> None:
    content = LAUNCHER_PATH.read_text(encoding="utf-8")

    console_index = content.index('"webots.exe"')

    wrapper_index = content.index('"webotsw.exe"')

    assert console_index != wrapper_index
    assert "webots-bin" in content
    assert "Get-RunningWebotsProcesses" in content
    assert "Start-WebotsCandidate" in content
    assert "RedirectStandardOutput" in content
    assert "RedirectStandardError" in content


def test_showcase_launcher_uses_valid_realtime_mode() -> None:
    content = LAUNCHER_PATH.read_text(encoding="utf-8")

    assert "--mode=realtime" in content
    assert "--mode=real-time" not in content
    assert "Stop-AllWebotsProcesses" in content
    assert "WEBOTS_PYTHON_COMMAND" in content
    assert "Get-RunningWebotsProcesses" in content
