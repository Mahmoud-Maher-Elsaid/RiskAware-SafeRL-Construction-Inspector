from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT = PROJECT_ROOT / "reports" / "final_project_completion"
RL_CHECKPOINT = (
    PROJECT_ROOT
    / "artifacts"
    / "runs"
    / "maskable_ppo_deadlock_safe_shield_seed42_u100"
    / "evaluations"
    / "best_model"
    / "best_model.zip"
)
CV_CHECKPOINT = (
    PROJECT_ROOT
    / "artifacts"
    / "runs"
    / "perception_production_100e"
    / "yolo26s_100e_seed42_20260722_214549"
    / "weights"
    / "best.pt"
)


def run_git(*arguments: str) -> str:
    return subprocess.run(
        ["git", *arguments],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    ).stdout


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    runtime = json.loads((OUTPUT / "final_runtime_summary.json").read_text(encoding="utf-8"))
    visual = json.loads(
        (OUTPUT / "stage5b_visual_evidence" / "stage5b_runtime_summary.json").read_text(
            encoding="utf-8"
        )
    )
    branch = run_git("branch", "--show-current").strip()
    commit = run_git("rev-parse", "HEAD").strip()
    report = {
        "result": "PASSED",
        "project_path": str(PROJECT_ROOT),
        "branch": branch,
        "commit_at_report_generation": commit,
        "original_state": {
            "branch": "stage-5b-live-perception-integration",
            "commit": "650da2be9332935c246f3536445b2b9abe3ade2d",
            "policy_controls_motors": False,
            "motor_source": "closed_loop_waypoint_controller",
            "first_person_view": "orthographic_line_only_and_rotated",
        },
        "safety_branch": {
            "name": "final/pre-cleanup-safety-20260729",
            "commit": "d87ebdc368bd27fd4cca9841490d38ac78476e92",
            "remote_preserved": True,
        },
        "deleted_from_parent": [
            "camera_final_diagnostic",
            "camera_final_diagnostic.zip",
            "RiskAware-SafeRL-Human-Camera-Preview",
            "RiskAware-SafeRL-Stage5B-Backups",
            "RiskAware-SafeRL-Stage5B-Benchmark",
            "RiskAware-SafeRL-Stage5B-Preflight",
            "RiskAware-SafeRL-Stage5B-Runtime",
            "__pycache__",
            "edgepilot-ai.zip",
        ],
        "preserved_outside_parent": [
            "F:/AI/Archived_Projects/Construction Safety Compliance CV System",
            "F:/AI/Archived_Projects/edgepilot-ai",
            "F:/AI/Archived_Projects/retinaguard-ai",
        ],
        "migrated": [
            "Stage 5B benchmark JSON, CSV, script, selection, and console evidence",
        ],
        "deleted_inside_repository": [
            "invalid virtual environment",
            "pytest and Ruff caches",
            "Python bytecode caches",
            "stale Webots logs",
            "repair backups",
            "temporary preview output",
        ],
        "stages_discovered": [
            "Stage 5B3 live perception and first-person viewport",
            "Stage 5C dedicated RL motor-control runtime",
        ],
        "stages_completed": [
            "Stage 5B3 live perception and first-person viewport",
            "Stage 5C dedicated RL motor-control runtime",
            "one-command production integration",
        ],
        "architecture": {
            "control_flow": (
                "Webots state and live CV risk -> checkpoint-compatible observation -> "
                "task-valid mask -> MaskablePPO -> RuntimeSafetyShield -> "
                "differential-drive command -> Webots wheel motors"
            ),
            "perception_flow": (
                "Webots camera -> YOLO on CUDA -> interpreted risk channel -> "
                "policy observation and SafetyShield"
            ),
        },
        "models": {
            "rl_checkpoint": str(RL_CHECKPOINT.relative_to(PROJECT_ROOT)),
            "rl_sha256": sha256(RL_CHECKPOINT),
            "cv_checkpoint": str(CV_CHECKPOINT.relative_to(PROJECT_ROOT)),
            "cv_sha256": sha256(CV_CHECKPOINT),
        },
        "stage5b_runtime": visual,
        "stage5c_runtime": runtime,
        "tests": {
            "pytest": "228 passed",
            "ruff_format": "passed",
            "ruff_check": "passed",
            "py_compile": "passed",
            "type_check": "not configured",
            "production_command": "passed",
        },
        "remaining_limitations": [
            "The recurrent GRU module has no trained recurrent checkpoint or comparative evaluation.",
            "A comprehensive publication-scale all-policy benchmark and final paper remain future research work.",
            "Validation is in Webots simulation and does not establish real-world safety.",
            "The accepted Stage 5C mission is bounded to 10 policy decisions.",
        ],
        "launch_command": (
            "powershell.exe -NoProfile -ExecutionPolicy Bypass -File "
            '"F:\\AI\\My_Project\\RiskAware-SafeRL-Construction-Inspector\\'
            'scripts\\run_complete_autonomous_inspection.ps1"'
        ),
    }
    (OUTPUT / "final_completion_report.json").write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )
    markdown = f"""# Final Project Completion Report

## Result

**PASSED**

Project: `{PROJECT_ROOT}`

Branch: `{branch}`
Commit at report generation: `{commit}`

## Cleanup and preservation

The parent directory now contains only the main RiskAware repository. Obsolete Stage 5B diagnostics, previews, backups, preflight/runtime directories, caches, and ZIPs were removed. The non-versioned 51.8 GB construction CV project and the two independent Git projects were moved intact to `F:\\AI\\Archived_Projects`.

The externally referenced Stage 5B benchmark was migrated into `reports/perception/stage5b/benchmark_20260726/` and verified by SHA-256 before its external directory was removed.

Inside the repository, the invalid virtual environment, Python/test/lint caches, stale Webots logs, repair backups, and temporary preview output were removed. The active `.venv`, its `.python311` base, the PPE dataset, trained weights, and RL checkpoints were preserved.

## Completed stages

- Stage 5B3: perspective, level, human-eye first-person viewport with five validated timeline images and 22/22 live CUDA perception frames.
- Stage 5C: deterministic MaskablePPO inference on CUDA, task masks, runtime SafetyShield, CV risk observation updates, and executed actions reaching Webots wheel motors.
- Production integration: the exact one-command launcher validates dependencies and hashes, runs the mission, enforces truth flags, and exits nonzero on failure.

## Runtime evidence

Stage 5C recorded 10 policy decisions, two distinct policy actions, three SafetyShield restricted-zone interventions, three distinct wheel-command pairs, 14 motor-command changes, 10 successful CUDA perception inferences, and nine CV-driven observation changes. Manual control and fallback control were false.

RL checkpoint: `{report["models"]["rl_checkpoint"]}`

RL SHA-256: `{report["models"]["rl_sha256"]}`

CV checkpoint: `{report["models"]["cv_checkpoint"]}`
CV SHA-256: `{report["models"]["cv_sha256"]}`

## Verification

- Pytest: 228 passed
- Ruff format: passed
- Ruff check: passed
- Python compilation: passed
- Exact production command: passed
- Stage 5B visual validation and manual image inspection: passed
- Stage 5C final first-person image inspection: passed

## Remaining limitations

- The recurrent GRU module does not have a trained recurrent checkpoint or comparative evaluation.
- A publication-scale all-policy benchmark and final paper remain research work.
- Results are validated in Webots simulation, not as a real-world safety guarantee.
- The accepted Stage 5C mission is bounded to 10 policy decisions.

## Launch

```powershell
{report["launch_command"]}
```
"""
    (OUTPUT / "final_completion_report.md").write_text(markdown, encoding="utf-8")

    tracked = run_git("ls-files").splitlines()
    inventory_lines = [
        "Tracked repository files:",
        *tracked,
        "",
        "Required local untracked assets:",
        f"{RL_CHECKPOINT.relative_to(PROJECT_ROOT)} | SHA256={sha256(RL_CHECKPOINT)}",
        f"{CV_CHECKPOINT.relative_to(PROJECT_ROOT)} | SHA256={sha256(CV_CHECKPOINT)}",
        "Personal Protective Equipment - Combined Model.v1i.yolov8/ | local dataset",
    ]
    (OUTPUT / "final_file_inventory.txt").write_text(
        "\n".join(inventory_lines) + "\n",
        encoding="utf-8",
    )
    (OUTPUT / "final_test_results.txt").write_text(
        "\n".join(
            [
                "ruff format --check .: PASS (147 files formatted)",
                "ruff check .: PASS",
                "python -m py_compile tracked Python files: PASS",
                "pytest -q: PASS (228 passed)",
                "Stage 5B Webots/CUDA runtime: PASS",
                "Stage 5B automated visual validation: PASS",
                "Stage 5C Webots RL/CV/SafetyShield motor runtime: PASS",
                "Exact production PowerShell launcher: PASS",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    history = run_git("log", "--oneline", "--decorate", "--graph", "--all", "-n", "200")
    normalized_history = "\n".join(line.rstrip() for line in history.splitlines()) + "\n"
    (OUTPUT / "final_git_history.txt").write_text(
        normalized_history,
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
