# Final Stage Completion Matrix

The final research roadmap is separate from the historical repository stage
numbers. This matrix is updated only after implementation and validation.

| Final stage | Historical mapping | Initial status | Completion gate |
|---|---|---|---|
| 0 — Foundation | Stage 0 plus accumulated documentation/CI | PARTIAL | Documentation, links, static checks, tests, production regression |
| 1 — Grid benchmark | Historical Stages 1–2 | VERIFIED_COMPLETE | Gymnasium checker and deterministic scenario suite passed |
| 2 — Expert planners | Historical Stages 3A–3B | VERIFIED_COMPLETE | Three planners produced 90 real evaluation records |
| 3 — PPO and SAC | Historical Stages 2D–3D | VERIFIED_COMPLETE | Both genuine checkpoints loaded; 60 deterministic evaluations completed |
| 4 — RiskShield-PPO | Historical Stages 3E-B/3E-C | PARTIAL | Constrained objective affects training and comparisons execute |
| 5 — Predictive shield | Historical Stage 3E and Stage 5C shield | PARTIAL | Structured k-step decisions and benchmark pass |
| 6 — Three Webots worlds | Historical Stages 4A–5C | PARTIAL | Three worlds load, move, render, and shut down cleanly |
| 7 — Perception | Historical Stage 5B plus Stage 5C integration | PARTIAL | Live typed detection changes semantic runtime state |
| 8 — Uncertainty | Domain-randomization utility | MISSING | Required deterministic conditions produce raw/summary results |
| 9 — Benchmark | Existing isolated evaluations | MISSING | Exactly 1,350 unique primary runs plus ablations and statistics |
| 10 — Paper/release | Paper skeleton | PARTIAL | Paper values derive from outputs; final acceptance and release pass |

The accepted Stage 5C path remains a protected regression baseline throughout
this work. New research components do not replace its motor-control path unless
regression evidence proves compatibility.
