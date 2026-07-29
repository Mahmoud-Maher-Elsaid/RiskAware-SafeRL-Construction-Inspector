# Project Completion Plan

Last updated: 2026-07-29

## Live checklist

| Field | Current value |
|---|---|
| Current task | Final acceptance, reporting, and Git integration |
| Current state | Stage 5B first-person CUDA perception and Stage 5C RL motor-control runtime passed |
| Evidence collected | Webots R2025a; PyTorch `2.11.0+cu128`; CUDA 12.8; RTX 3070 Ti; verified RL/CV hashes; curated visual, action, shield, motor, and perception evidence |
| Files changed | Stage 5 builders, worlds, controllers, launcher, validators, tests, configs, perception, policy gate, recurrent/domain modules, CI, README, paper, and docs |
| Tests run | Ruff lint/format; 228-test full pytest suite; Python compilation; real Stage 5B and Stage 5C Webots/CUDA runs; exact production launcher |
| Runtime result | Stage 5B: 8/8 deterministic waypoints, 22/22 CV frames, valid first-person views. Stage 5C: 10 RL decisions, three shield interventions, 14 motor-command changes, 10/10 CUDA CV inferences |
| Remaining work | Trained recurrent-policy comparison, comprehensive all-policy publication benchmark, and paper completion |
| Blockers | No trained recurrent checkpoint or approved publication experiment budget/protocol exists; these are research expansions beyond the named Stage 5C runtime gate |

## Phase checklist

- [x] Preserve current tracked, untracked, ignored runtime, and repair-backup state externally.
- [x] Inspect branch, HEAD, top-level tree, dependency definitions, CI, Stage 5 assets, route, and launcher baseline.
- [x] Create permanent repository working rules.
- [x] Recover Python 3.11 environment and record baseline Ruff/pytest/model checks.
- [x] Diagnose and repair Stage 5A3 launcher, physics, calibration, navigation, camera, and validation.
- [x] Verify Stage 5A and Stage 5A3 in real Webots; repeat Stage 5A3 three times.
- [x] Integrate Stage 5 sources cleanly and verify deterministic world generation.
- [x] Implement and test truthful modular semantic perception.
- [x] Validate checkpoint and implement gated safe policy integration.
- [x] Complete the named implementation roadmap through the dedicated Stage 5C policy-motor runtime.
- [x] Preserve existing reproducible benchmark JSON/CSV artifacts and migrate the Stage 5B benchmark into the repository.
- [x] Complete Ruff, full pytest, CI/test markers, documentation, and paper alignment.
- [x] Commit and push stage-specific completion work.

## Safety evidence

- Remote branch: `final/pre-cleanup-safety-20260729`
- Safety commit: `d87ebdc368bd27fd4cca9841490d38ac78476e92`
- The remote branch preserves the legitimate pre-cleanup Stage 5B working state.

## Initial state notes

- Source branch: `stage-5b-live-perception-integration` at `650da2b`.
- Pre-cleanup work contained legitimate Stage 5B viewpoint repairs plus ignored caches, environments, checkpoints, datasets, and runtime evidence.
- The original main viewport was orthographic, line-only, and rotated; static tests did not detect the visual failure.
- The Webots executable accepts the world as a positional argument, and supports `--batch`, `--minimize`, `--no-rendering`, `--stdout`, and `--stderr`.
- The original verified motor source was the deterministic Stage 5A3 waypoint controller, and RL motor control was false.
