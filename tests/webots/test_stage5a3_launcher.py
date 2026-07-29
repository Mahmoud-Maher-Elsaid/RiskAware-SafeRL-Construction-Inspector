from __future__ import annotations

import importlib.util
from pathlib import Path

RUNNER_PATH = Path("scripts/run_stage5a3_closed_loop_mission.py")


def load_runner():
    specification = importlib.util.spec_from_file_location("stage5a3_runner", RUNNER_PATH)
    assert specification and specification.loader
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def test_world_argument_is_a_distinct_final_argument() -> None:
    runner = load_runner()
    world = Path(r"F:\project path\webots\worlds") / runner.WORLD_NAME
    arguments = runner.build_webots_arguments(world, "validation")
    assert arguments[-1] == str(world)
    assert "--batch" in arguments
    assert "--no-rendering" in arguments


def test_interactive_arguments_keep_rendering_visible() -> None:
    runner = load_runner()
    arguments = runner.build_webots_arguments(Path(r"F:\world.wbt"), "interactive")
    assert "--batch" not in arguments
    assert "--no-rendering" not in arguments


def test_world_title_validation_rejects_empty_world() -> None:
    runner = load_runner()
    assert runner.validate_world_title(f"Webots - {runner.WORLD_NAME}")
    assert not runner.validate_world_title("Webots - empty.wbt")
    assert not runner.validate_world_title("Webots")
