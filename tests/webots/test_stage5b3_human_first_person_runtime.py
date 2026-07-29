from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

RUNNER_PATH = PROJECT_ROOT / "scripts" / "run_verified_human_first_person_stage5b3.ps1"

LAUNCHER_PATH = PROJECT_ROOT / "scripts" / "run_stage5a3_closed_loop_mission.py"


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
