# Final Research Roadmap Audit

Audit baseline: `main` commit `40dee1e`, with the production regression rerun on
`final/research-benchmark-release` on 2026-07-29.

The repository's historical stage labels describe its development sequence and
do not match the final research roadmap. The mapping below is evidence-based.

| Final stage | Existing repository work | Audit status | Remaining work |
|---|---|---|---|
| 0 — Foundation | Packaging, CI, README, Git history, release tag | PARTIAL | Final architecture, methodology, reproducibility, demo, troubleshooting, changelog, link validation |
| 1 — Grid benchmark | Research environment, scenarios, masks, semantic channels, tests | VERIFIED_COMPLETE | None |
| 2 — Expert planners | Risk-aware A*, frontier exploration, nearest-risk revisit | VERIFIED_COMPLETE | None |
| 3 — PPO and SAC | Genuine PPO and SAC training, checkpoints, deterministic evaluation | VERIFIED_COMPLETE | Learned-policy mission success is a documented negative result |
| 4 — RiskShield-PPO | PPO-Lagrangian, learned cost value, multiplier telemetry | VERIFIED_COMPLETE | Multiplier saturation and zero mission success remain limitations |
| 5 — Predictive shield | Structured k-step research shield and Stage 5C motor-loop shield | VERIFIED_COMPLETE | None for simulator scope |
| 6 — Three Webots worlds | Parameterized small, medium, and dynamic worlds | VERIFIED_COMPLETE | None for simulation gate |
| 7 — Perception | Live YOLO CUDA inference and typed semantic risk mapping | VERIFIED_COMPLETE | Detector classes and real-world validity remain limited |
| 8 — Uncertainty | Deterministic false-negative, visual, density, and layout perturbations | VERIFIED_COMPLETE | 320 unique episodes completed |
| 9 — Research benchmark | Five algorithms across 27 conditions and ten paired seeds | VERIFIED_COMPLETE | 1,350 unique primary runs and 180 ablations completed with no unresolved failures |
| 10 — Paper and release | Skeleton IEEE LaTeX file | PARTIAL | Real results, verified references, complete sections, release/acceptance tooling |

## Verified baseline facts

- The production command completed with exit code zero.
- `policy_loaded`, `policy_controls_motors`, `safety_shield_active`,
  `cv_model_connected`, `cuda_inference_verified`,
  `perception_live_during_mission`, and
  `perception_affects_runtime_state` are true in the accepted runtime summary.
- `manual_control_used` and `fallback_controller_used` are false.
- The production RL checkpoint is a genuine 102,400-step MaskablePPO checkpoint.
- RiskShield-PPO's constrained objective changed training, but its multiplier
  saturated and it did not solve complete missions.
- PPO, SAC, and RiskShield-PPO all achieved zero full-mission success in the
  final grid matrix; planner success was substantially stronger.
- The Stage 9 matrix contains 1,350 unique primary records, zero missing runs,
  zero unresolved failures, and 180 paired ablation records.
- Stage 10 remains incomplete until the paper and release gates pass.

## Classification vocabulary

`VERIFIED_COMPLETE` means source, tests, entry point, and runtime evidence agree.
`IMPLEMENTED_NOT_RUNTIME_VERIFIED` means implementation exists without applicable
runtime evidence. `PARTIAL` means only a subset is implemented. `DRY_RUN_ONLY`,
`MOCKED`, `MISSING`, `OBSOLETE`, `BROKEN`, and `NOT_APPLICABLE` retain their
literal meanings.
