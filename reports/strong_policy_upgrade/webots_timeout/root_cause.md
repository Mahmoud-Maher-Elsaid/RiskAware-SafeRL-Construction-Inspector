# Webots Timeout Root-Cause Analysis

## Findings

The timeout had three sequential, independently observed causes:

1. `Start-Process` could not materialize the child environment when both case variants `Path` and `PATH` were inherited. This failed before Webots startup.
2. After environment normalization, the live Ultralytics backend attempted to write its settings file under the user profile and failed with `PermissionError: [WinError 5]`. The child now uses the repository-local `YOLO_CONFIG_DIR`.
3. The unattended Webots child used Qt's `offscreen` backend. In the Stage 5C world this allowed startup but did not advance to the completion marker. The launcher now uses the Windows Qt backend with `--batch --no-rendering --minimize`, bounded scaling variables, and a process-tree cleanup helper.

## Repair validation

`run_final_world_smoke.ps1 -TimeoutSeconds 60` completed all three generated worlds. The production command then completed with exit code 0 and produced `webots/logs/stage5c_rl_motor_runtime/stage5c_runtime_summary.json` and `reports/final_project_completion/final_runtime_summary.json`.

The validated production summary reports live CUDA perception, policy motor control, Safety Shield activity, no manual control, no fallback controller, and `mission_completed: true`.

## Scope

The repair is child-process scoped. It does not modify machine-wide display settings or the user environment. Historical failed logs and evidence remain preserved.
