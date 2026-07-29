# Final Research Roadmap Audit

Audit baseline: `main` commit `40dee1e`, with the production regression rerun on
`final/research-benchmark-release` on 2026-07-29.

The repository's historical stage labels describe its development sequence and
do not match the final research roadmap. The mapping below is evidence-based.

| Final stage | Existing repository work | Audit status | Remaining work |
|---|---|---|---|
| 0 — Foundation | Packaging, CI, README, Git history, release tag | PARTIAL | Final architecture, methodology, reproducibility, demo, troubleshooting, changelog, link validation |
| 1 — Grid benchmark | `ConstructionInspectionEnv`, scenarios, masks, semantic channels, tests | PARTIAL | Orientation, dynamic hazards, PPE risk, configurable noise, typed schema, configs, serialization and checker gate |
| 2 — Expert planners | A*, oracle inspection, baseline evaluator | PARTIAL | Frontier and nearest-risk planners, risk-aware cost, replanning telemetry, final evaluation |
| 3 — PPO and SAC | Genuine PPO/MaskablePPO training and checkpoints | PARTIAL | SAC adapter/training/checkpoint, unified deterministic evaluation and final plots |
| 4 — RiskShield-PPO | Lagrange multiplier and prior constrained experiments | PARTIAL | Benchmark-ready constrained training path, cost telemetry, comparisons, documented objective |
| 5 — Predictive shield | One-step semantic `SafetyShield`; Stage 5C motor-loop shield | PARTIAL | Structured k-step prediction, moving-worker prediction, latency and shield ablations |
| 6 — Three Webots worlds | Six historical worlds; verified Stage 5C world | PARTIAL | Exactly three parameterized final worlds and smoke/visual/physics evidence |
| 7 — Perception | Live YOLO CUDA inference, temporal state integration, model hash | PARTIAL | Final typed semantic-risk mapper, model card, stale-frame and projection validation |
| 8 — Uncertainty | Basic domain randomization only | MISSING | Deterministic FN, visual, density, unseen-layout protocol and experiments |
| 9 — Research benchmark | Historical scenario evaluations | MISSING | Resumable 1,350-run matrix, statistics, figures, ablations |
| 10 — Paper and release | Skeleton IEEE LaTeX file | PARTIAL | Real results, verified references, complete sections, release/acceptance tooling |

## Verified baseline facts

- The production command completed with exit code zero.
- `policy_loaded`, `policy_controls_motors`, `safety_shield_active`,
  `cv_model_connected`, `cuda_inference_verified`,
  `perception_live_during_mission`, and
  `perception_affects_runtime_state` are true in the accepted runtime summary.
- `manual_control_used` and `fallback_controller_used` are false.
- The production RL checkpoint is a genuine 102,400-step MaskablePPO checkpoint.
- Existing constrained runs contain Lagrange metadata and telemetry, but the final
  RiskShield-PPO method has not yet passed the requested comparisons.
- No SAC checkpoint or full research matrix exists at audit time.
- The paper is a section skeleton and its bibliography contains no references.

## Classification vocabulary

`VERIFIED_COMPLETE` means source, tests, entry point, and runtime evidence agree.
`IMPLEMENTED_NOT_RUNTIME_VERIFIED` means implementation exists without applicable
runtime evidence. `PARTIAL` means only a subset is implemented. `DRY_RUN_ONLY`,
`MOCKED`, `MISSING`, `OBSOLETE`, `BROKEN`, and `NOT_APPLICABLE` retain their
literal meanings.
