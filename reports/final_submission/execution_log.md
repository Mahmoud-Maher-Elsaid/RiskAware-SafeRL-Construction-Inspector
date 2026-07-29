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

## 2026-07-29T17:20:00Z — Stage 5 predictive Safety Shield

- Stage: 5
- Commands:
  - `.venv\Scripts\python.exe -m pytest -q tests\safety\test_predictive_safety_shield.py`
  - `.venv\Scripts\python.exe scripts\benchmark_predictive_shield.py`
- Decision: keep the research k-step shield isolated from the verified
  production Stage 5C implementation.
- Result: unit, boundary, moving-hazard, k-step, and emergency-stop tests passed.
  Disabled, one-step, and three-step modes completed 90 deterministic episodes.
- Evidence:
  - `reports/final_submission/stage5_safety_shield/raw_results.csv`
  - `reports/final_submission/stage5_safety_shield/summary.json`

## 2026-07-29T17:45:00Z — Stage 6 final Webots environments

- Stage: 6
- Commands:
  - `.venv\Scripts\python.exe scripts\build_final_webots_worlds.py`
  - `powershell.exe -NoProfile -ExecutionPolicy Bypass -File scripts\run_final_world_smoke.ps1`
  - `.venv\Scripts\python.exe scripts\validate_final_world_visuals.py`
  - `.venv\Scripts\python.exe -m pytest -q tests\webots\test_final_worlds.py`
- Failure: the first harness stopped Webots after the completion marker but
  before final viewport export.
- Repair: add a post-marker evidence grace period matching production.
- Failure: the first small-world footprint exposed a sky-colored void beneath
  perimeter hoarding; inherited edge and horizon checks also misclassified
  low-poly perspective geometry.
- Repair: use grounded 24×18, 26×20, and 28×22 m slabs, add construction
  hoarding, validate roll from vertical structures, and document the calibrated
  low-poly edge threshold.
- Result: all three worlds completed real RL/CV autonomous missions with clean
  controller startup and shutdown. Automated visual checks and manual inspection
  passed for all three final frames.
- Evidence:
  - `reports/final_submission/stage6_webots/world_smoke_summary.json`
  - `reports/final_submission/stage6_webots/visual_validation.json`
  - per-world runtime summaries and first-person frames

## 2026-07-29T18:00:00Z — Stage 7 perception and semantic risk mapping

- Stage: 7
- Commands:
  - `.venv\Scripts\python.exe -m pytest -q tests\perception ...`
  - `.venv\Scripts\python.exe scripts\validate_final_perception.py`
  - `Get-FileHash -Algorithm SHA256 <production checkpoint>`
- Failure: a test compared a `float32` risk value with exact decimal equality.
- Repair: use a numerical tolerance without changing runtime calculations.
- Result: 22 perception tests passed. The production checkpoint hash matched;
  CUDA inference, live mission perception, semantic state changes, ten
  annotations, and zero failures were verified from the accepted runtime.
- Evidence:
  - `reports/final_submission/stage7_perception/validation.json`
  - `reports/final_submission/stage7_perception/selected_runtime_annotation.png`

## 2026-07-29T18:20:00Z — Stage 8 uncertainty experiments

- Stage: 8
- Commands:
  - `.venv\Scripts\python.exe -m pytest -q tests\uncertainty`
  - `.venv\Scripts\python.exe scripts\run_uncertainty_experiments.py`
- Decision: perturb only agent observations while preserving clean simulator
  truth for metrics and shield prediction.
- Result: four tests passed. All 320 expected episodes completed with 320 unique
  run identifiers across 64 conditions. Mean robustness score was 0.8893 under
  the documented normalization; baseline mission success was zero and therefore
  contributes a neutral ratio rather than an invented improvement.
- Evidence:
  - `reports/final_submission/stage8_uncertainty/raw_results.csv`
  - `reports/final_submission/stage8_uncertainty/aggregated_results.csv`
  - `reports/final_submission/stage8_uncertainty/summary.json`

## 2026-07-29T19:00:00+03:00 — Interrupted Stage 9 recovery

- Stage: 9 recovery
- Commands:
  - `git status --short --branch`
  - `git status --porcelain=v2 -uall`
  - `git branch -a -vv`
  - `git log --oneline --decorate --graph -30`
  - `git diff --stat`
  - `git diff --cached --stat`
  - `git ls-files --others --exclude-standard`
  - `.venv\Scripts\python.exe -c "import pyarrow, scipy; ..."`
  - `.venv\Scripts\python.exe -m py_compile ...`
- Decision: preserve the coherent unfinished Stage 9 source and five valid
  risk-aware A* smoke-cache records. Continue in place without resetting,
  cleaning, stashing, or recreating the branch.
- Result: local and remote branch heads both resolve to `1059a4e4213de1b27a54ddc3a2a87e1557219f76`.
  No staged changes were present. PyArrow 21.0.0 and SciPy 1.17.1 imported
  successfully. The unfinished Python files compiled successfully.
- Resume point: harden the benchmark record/cache schema and recovery behavior,
  then execute the five-algorithm adapter gate before committing the harness.
- Evidence: `reports/final_submission/interruption_recovery.json`

## 2026-07-29T19:20:00+03:00 — Stage 9 benchmark harness gate

- Stage: 9
- Commands:
  - `.venv\Scripts\python.exe -m ruff check scripts\run_full_research_benchmark.py src\riskaware_saferrl\benchmarking tests\benchmarking`
  - `.venv\Scripts\python.exe -m pytest -q tests\benchmarking`
  - `.venv\Scripts\python.exe scripts\run_full_research_benchmark.py --validate-only`
  - `.venv\Scripts\python.exe scripts\run_full_research_benchmark.py --one-per-algorithm --output reports\final_submission\stage9_benchmark\validation\five_algorithm_gate`
- Decision: make run identifiers immutable with canonical configuration and
  checkpoint hashes; write per-run caches and resume state atomically; serialize
  GPU policy evaluation while parallelizing planners.
- Result: five benchmark tests passed. The configuration produced exactly 1,350
  unique manifest entries. One real episode completed through each of the five
  algorithm adapters, and a second invocation resumed all five records without
  rewriting cache files.
- Evidence:
  - `reports/final_submission/stage9_benchmark/validation/five_algorithm_gate/benchmark_summary.json`
  - `reports/final_submission/stage9_benchmark/validation/five_algorithm_gate/raw_results.csv`

## 2026-07-29T20:25:00+03:00 — Stage 9 full benchmark

- Stage: 9
- Commands:
  - `.venv\Scripts\python.exe scripts\run_full_research_benchmark.py`
  - `.venv\Scripts\python.exe scripts\run_stage9_ablations.py`
  - `.venv\Scripts\python.exe scripts\generate_final_figures.py`
  - `powershell.exe -NoProfile -ExecutionPolicy Bypass -File scripts\run_full_research_benchmark.ps1`
- Failure: SciPy returned a non-finite Wilcoxon p-value for paired success
  arrays that were identically zero.
- Root cause: PPO and RiskShield-PPO both had zero success in all paired cells,
  so the signed-rank statistic is undefined.
- Repair: explicitly assign p=1.0 only when every paired difference is zero and
  reject non-finite JSON values.
- Result: 1,350 expected and unique primary runs completed, with zero duplicate,
  missing, or unresolved failed runs. CSV and Parquet counts match. Six
  ablations completed 180 paired episodes. Thirteen figures and their source
  CSV files plus four final tables were generated from real records.
- Evidence:
  - `reports/final_submission/stage9_benchmark/benchmark_summary.json`
  - `reports/final_submission/stage9_benchmark/statistical_analysis.md`
  - `reports/final_submission/stage9_benchmark/ablation_summary.json`
  - `reports/final_submission/stage9_benchmark/figures`
