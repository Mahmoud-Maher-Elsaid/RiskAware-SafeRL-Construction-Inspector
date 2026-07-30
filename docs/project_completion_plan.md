# Project Completion Plan

Last updated: 2026-07-30

## Live checklist

| Field | Current value |
|---|---|
| Current task | Hierarchical recovery: RiskShield-Hierarchical-HRMPPO-MPC v4 |
| Current state | Exact recovery at 7bd5e656 passed; immutable datasets and v2/v3 checkpoints are preserved; the four-layer hierarchical implementation is in progress and v1 remains production |
| Evidence collected | Exact 5,460-count decomposition; immutable dataset/checkpoint hash verification; CUDA v3 smoke/resume; paired 90-episode initialization evaluation retained as negative evidence |
| Files changed | Versioned safety contract and migration; five-critic recurrent constrained PPO; vector-cost buffer; event-aware observation-only shield; evaluator/trainer; regression tests |
| Tests run | Repository Ruff and 338 tests before telemetry correction; focused safety/algorithm suite 27 tests after correction |
| Runtime result | Existing v1 production regression is blocked by a reproducible Webots R2025a/Qt invalid-framebuffer geometry failure; no v3 production claim has been made |
| Remaining work | Implement and validate hierarchical mission RL, causal local planning, predictive local control, hierarchical datasets/training, Webots launcher repairs, and every downstream acceptance gate |
| Blockers | No external blocker is currently established; flat v3 is retained only as negative evidence while the hierarchical recovery is implemented |

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
- [x] Recover and externally back up the interrupted strong-policy working tree.
- [x] Preserve and hash-verify the accepted 120,144 expert transitions.
- [x] Replace copied in-memory sequences with a two-chunk LRU mmap cache and compact index.
- [x] Pass the 1,000-batch CUDA Behavior Cloning memory regression gate.
- [ ] Pass Behavior Cloning held-out accuracy and per-action recall gates.
- [x] Implement and validate a policy-observation-only causal expert.
- [x] Pass causal expert solvability gates across 1,440 environment episodes.
- [x] Generate and hash-validate the separate causal correction dataset.
- [x] Compare causal-only, privileged-pretraining, and confidence-filtered modes.
- [x] Pass three-seed causal Behavior Cloning and CUDA memory gates.
- [x] Complete three genuine DAgger iterations.
- [ ] Train and accept RiskShield-HRMPPO v2 against all replacement gates.
- [x] Audit and decompose all 5,460 legacy constraint increments.
- [x] Implement Safety Contract v3 and preserve the legacy inactivity-biased diagnostic.
- [x] Implement typed vector-cost transitions and five independent recurrent cost critics.
- [x] Implement and regression-test the predictive event-aware shield.
- [x] Generate and validate 51,000 targeted safety DAgger corrections with
  contiguous recurrent context and KL task anchoring.
- [ ] Train and accept RiskShield-HRMPPO-Safe v3 against paired task and safety gates.
- [ ] Complete CV audit, three-world Webots validation, benchmark-v2, ablations, paper, and Git acceptance.

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
