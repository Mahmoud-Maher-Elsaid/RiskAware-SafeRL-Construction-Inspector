# Final Submission Execution Log

This log records commands, decisions, failures, repairs, validation evidence, and
commits for the final research benchmark release. Times use UTC.

## 2026-07-29T15:50:00Z — Initial regression and branch

- Stage: initial regression gate
- Commands:
  - `git status --short --branch`
  - `git branch -a -vv`
  - `git log --oneline --decorate --graph -20`
  - `git tag -n`
  - `powershell.exe -NoProfile -ExecutionPolicy Bypass -File scripts/run_complete_autonomous_inspection.ps1`
  - `.venv\Scripts\python.exe -m pytest -q`
- Decision: preserve Stage 5C as the production regression baseline and develop
  the research release on `final/research-benchmark-release`.
- Result: production runtime passed with Webots R2025a and CUDA on an NVIDIA
  GeForce RTX 3070 Ti Laptop GPU. All 228 existing tests passed.
- Evidence:
  - `reports/final_project_completion/final_runtime_summary.json`
  - `webots/logs/stage5c_rl_motor_runtime/stage5c_runtime_summary.json`
- Branch: `final/research-benchmark-release`, created from `40dee1e`.

## 2026-07-29T15:55:00Z — Final-roadmap audit

- Stage: Phase 1
- Commands:
  - `rg --files src configs scripts tests paper`
  - `rg -n '(TODO|FIXME|TBD|mock|dry.run|not implemented|missing)'`
  - checkpoint, metadata, Webots-world, and paper inventory commands
- Decision: map the final research roadmap independently from the repository's
  historical numbering. Reuse verified components; extend rather than replace
  the production Stage 5C path.
- Result: Stages 0, 1, 2, 3, 5, 6, and 7 are partial; Stage 4 has experimental
  constrained components but lacks a final benchmark-ready method; Stages 8–10
  are incomplete.
- Evidence:
  - `reports/final_submission/stage_audit.md`
  - `reports/final_submission/stage_audit.json`
  - `docs/final_stage_completion_matrix.md`

## 2026-07-29T16:10:00Z — Stage 1 grid benchmark

- Stage: 1
- Commands:
  - `.venv\Scripts\python.exe -m pytest -q tests\envs\test_research_grid_environment.py`
  - `.venv\Scripts\python.exe scripts\validate_grid_environment.py`
  - `.venv\Scripts\python.exe -m ruff check ...`
- Decision: preserve the historical checkpoint-compatible environment and add a
  research extension with a typed, richer observation schema.
- Failure: the first isolated collision test inherited randomized risks, and the
  false-negative validation bound rejected a deliberate 100% dropout test.
- Root cause: incomplete fixture isolation and a shared density bound.
- Repair: clear unrelated risks in the fixture and validate perception dropout
  independently over `[0, 1]`.
- Result: seven targeted tests passed; all three configurations passed the
  Gymnasium, determinism, space, mask, and serialization validator.
- Evidence:
  - `reports/final_submission/stage1_grid_environment/validation.json`

## 2026-07-29T16:25:00Z — Stage 2 planner baselines

- Stage: 2
- Commands:
  - `.venv\Scripts\python.exe -m pytest -q tests\baselines\test_final_planners.py`
  - `.venv\Scripts\python.exe scripts\evaluate_planner_baselines.py`
- Failure: the first 90-episode evaluation exceeded the five-minute timeout.
- Root cause: a separate A* search ran for every candidate viewpoint on every step.
- Repair: replace repeated searches with one deterministic multi-target
  uniform-cost search and bounded nearest-first replanning.
- Result: seven planner tests passed. The optimized 90-episode evaluation
  completed in 3.67 seconds with 90 unique runs. Risk-aware A* and nearest-risk
  revisit achieved 30/30 successes; frontier exploration achieved 28/30.
- Evidence:
  - `reports/final_submission/stage2_planner_baselines/raw_results.csv`
  - `reports/final_submission/stage2_planner_baselines/summary.json`

## 2026-07-29T16:50:00Z — Stage 3 PPO and SAC baselines

- Stage: 3
- Commands:
  - `.venv\Scripts\python.exe scripts\train_ppo_research.py`
  - `.venv\Scripts\python.exe scripts\train_sac.py`
  - `.venv\Scripts\python.exe scripts\evaluate_rl_baselines.py`
- Failure: the first evaluator passed PPO's NumPy action array directly to the
  discrete environment.
- Root cause: the training vector wrapper normalized the action type, but the
  direct deterministic evaluation environment did not.
- Repair: explicitly convert PPO predictions to scalar discrete actions while
  retaining the continuous SAC array.
- Result: PPO trained for 30,208 steps and SAC for 10,000 steps on CUDA. Both
  checkpoints loaded and completed 60 deterministic cross-site episodes.
  Neither solved the full task; the negative results are retained.
- Evidence:
  - `reports/final_submission/stage3_rl_baselines/summary.json`
  - `reports/final_submission/stage3_rl_baselines/evaluation_results.csv`
  - `reports/final_submission/stage3_rl_baselines/training_metrics.csv`

## 2026-07-29T17:10:00Z — Stage 4 RiskShield-PPO

- Stage: 4
- Commands:
  - `.venv\Scripts\python.exe -m pytest -q tests\algorithms\test_riskshield_ppo.py`
  - `.venv\Scripts\python.exe scripts\train_riskshield_ppo.py`
  - `.venv\Scripts\python.exe scripts\evaluate_riskshield_ppo.py`
- Decision: implement PPO-Lagrangian with a learned auxiliary cost-value
  function. Keep the predictive shield separable and disabled during training.
- Result: the constraint altered the optimized reward, the cost critic updated,
  and the multiplier rose from zero to its cap. Early stopping activated at
  15,300 steps after 50 non-improving episodes. Four configurations completed
  120 deterministic evaluation episodes.
- Negative result: no learned policy completed the full inspection task.
  RiskShield-PPO reduced unshielded mean safety cost from PPO's 55.13 to 27.93;
  the k-step shield reduced both policies to approximately six cost units and
  eliminated observed collisions in this evaluation.
- Evidence:
  - `reports/final_submission/stage4_riskshield_ppo/comparison_results.csv`
  - `reports/final_submission/stage4_riskshield_ppo/comparison_summary.json`
  - `reports/final_submission/stage4_riskshield_ppo/constraint_training_metrics.csv`
