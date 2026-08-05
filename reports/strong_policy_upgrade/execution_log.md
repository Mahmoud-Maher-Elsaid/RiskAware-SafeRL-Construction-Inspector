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
- Commit: `af59974`

## 2026-07-29 — V2 environment and objective

- Decision: preserve `ResearchConstructionEnv` unchanged and add a versioned `ResearchConstructionEnvV2`
- Repair: persistent semantic evidence, explicit action-mask observation, cyclic orientation, recurrent context fields, and stable episode-level perception sampling
- Repair: task reward and safety cost are independent; progress uses discounted potential shaping
- Repair: inspection coverage and explored-area coverage are separate metrics
- Validation: original and v2 environment suites passed (14 tests)
- Validation: Gymnasium checker and deterministic step/reset checks passed
- Commit: `32457f7`

## 2026-07-29 — Expert demonstrations

- Generator: `scripts/generate_expert_demonstrations.py`
- Failed attempt 2: 120,099 transitions, 64.4% attempted success; rejected because shield outputs contradicted masks
- Failed attempt 3: 120,142 transitions, 76.8% attempted success; rejected because incomplete episodes entered the data
- Repair: episode-atomic admission retains only complete successful expert trajectories
- Final dataset: 120,144 transitions from 200 deterministic seeds
- Stored trajectory success: 100%
- Invalid stored actions: 0
- Failed-episode transitions rejected: 144,400
- Dataset SHA-256: `06f872d819241563d3bf735d97c203d3de9633fbbc9cb3e5e99f95a1edbb1285`
- Large chunks: retained locally under ignored `artifacts/strong_policy_upgrade/expert_demonstrations/chunks/`
- Reproducibility metadata: tracked manifest and dataset summary
