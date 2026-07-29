# Baseline Audit

Date: 2026-07-29  
Completion branch: `final/project-completion`  
Baseline commit: `d87ebdc`

## Result

The truthful baseline is **partial**. The repository has a strong tested simulation, training, semantic-safety, Webots bridge, deterministic Stage 5A3 navigation, and CUDA perception foundation. It does not yet have verified RL-to-Webots motor control, CV-driven runtime safety/navigation state, final visual acceptance evidence, or the requested production launcher.

## Environment

- Python 3.11.0 from `.venv/Scripts/python.exe`
- PyTorch 2.11.0 with CUDA 12.8
- CUDA available on NVIDIA GeForce RTX 3070 Ti Laptop GPU
- Stable-Baselines3 and SB3-Contrib 2.9.0
- Ultralytics 8.4.103
- Webots R2025a
- Local 14-class PPE dataset available and intentionally ignored by Git

## Models

- RL candidate: `artifacts/runs/maskable_ppo_deadlock_safe_shield_seed42_u100/evaluations/best_model/best_model.zip`
- RL SHA-256: `172437CAE45B69031F443C0707FB0795D2F1860D3B95594BE281645D8A173FE7`
- CV candidate SHA-256: `4BD2190A3C99FFA5D1A7F57A37683C908C044E91485CD01E44CA4E199BDE8550`

Both artifacts exist. The CV model has prior CUDA runtime evidence. The RL artifact has not yet been accepted as Webots-motor compatible in a live runtime.

## Verification

- Python compilation: passed.
- Pytest collection: 207 tests collected.
- Full pytest baseline: 207 passed.
- Ruff initially found one import-order/format issue in the newly migrated benchmark script; it was repaired.
- No type checker is configured in `pyproject.toml`.

## Implementation classification

| Area | Baseline classification |
|---|---|
| Stage 5A camera acquisition | Implemented; prior runtime evidence exists |
| Stage 5A3 navigation | Deterministic waypoint control; not reinforcement learning |
| Stage 5B perception | Connected and previously CUDA-verified; current acceptance rerun pending |
| Human first-person viewport | Partially implemented; visual acceptance pending |
| RL policy inference | Dry run only |
| RL motor control | Missing |
| SafetyShield | Implemented for the grid/policy pipeline; not yet verified in the Webots motor loop |
| CV influence on runtime decisions | Missing |
| Production launcher | Missing |

The baseline does not claim that RL controls motors or that perception affects navigation.
