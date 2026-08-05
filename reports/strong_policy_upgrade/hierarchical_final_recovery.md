# Hierarchical Final Recovery

Status: **PASSED**

The repository was recovered without changing any accepted artifact. Local and
remote `final/strong-policy-upgrade` both resolve to
`7bd5e656a9c52fce5e00ff826e1452223da15151`, and the working tree was clean
before this recovery record was added.

## Preserved evidence

- Privileged demonstrations: 120,144 transitions, dataset SHA-256
  `06f872d819241563d3bf735d97c203d3de9633fbbc9cb3e5e99f95a1edbb1285`.
- Systematic causal demonstrations: 250,128 transitions, dataset SHA-256
  `5278452907ed0f7a3ad1c4381caad643d9c03b74b3bff32747735223f6c7756f`.
- HRMPPO v2 task checkpoint SHA-256:
  `f296c5633ec6d3e9918474039ae079fb681e7dded02a4962c465010e2bf80436`.
- Failed flat HRMPPO-Safe v3 checkpoint SHA-256:
  `4ccb08bad71c0772d55b8a0a6f379f8b1189c56cde2150daa579c205fba9c0c1`.
- Previous negative reports remain available and production has not been
  replaced.

## Software validation

Python compilation, Ruff linting, Ruff formatting, and the complete 344-test
suite passed with the repository Python 3.11 environment.

## Recovery point

The next implementation stage is a new four-layer hierarchical system:
learned recurrent mission-option selection, observation-only causal local
planning, predictive local trajectory control, and an independent final Safety
Shield. The failed flat v3 architecture remains a preserved baseline and will
not be continued as the primary solution.

`HIERARCHICAL_FINAL_RECOVERY=PASSED`
