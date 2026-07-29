# Strong Policy Upgrade Execution Log

## 2026-07-29 — Regression baseline and failure diagnosis

- Branch: `final/strong-policy-upgrade`
- Baseline commit: `7e2d7c86adcd0533f59a020bfd40dfe7c30684f6`
- Production command result: `COMPLETE_AUTONOMOUS_INSPECTION=PASSED`
- Final acceptance result: `FINAL_ACCEPTANCE=PASSED`
- Baseline test result: 272 tests passed
- Expert-solvability command: `.venv\Scripts\python.exe scripts\evaluate_planner_baselines.py --seeds 100 --output-dir reports\strong_policy_upgrade\diagnosis_planners`
- Expert-solvability result: 900 real episodes; best success was 100% for small, medium, and dynamic sites
- Policy-diagnosis command: `.venv\Scripts\python.exe scripts\diagnose_policy_v1.py`
- Policy-diagnosis result: 180 direct checkpoint evaluations completed on CUDA
- Failure: PPO, SAC, and RiskShield-PPO v1 achieved zero mission success in the existing benchmark
- Root causes: unused action masks, flattened spatial inputs, missing recurrent memory, insufficient training, saturated Lagrange multiplier, duplicated safety penalties, and non-mission-aware early stopping
- Evidence: `failure_diagnosis.json`, `failure_diagnosis.md`, `v1_diagnostic_traces.json`, and `diagnosis_planners/`
