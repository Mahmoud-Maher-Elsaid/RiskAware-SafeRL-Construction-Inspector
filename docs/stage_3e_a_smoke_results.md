# Stage 3E-A Semantic Safety Shield Smoke Results

## Status

The refined semantic safety shield passed the complete test suite and CUDA
MaskablePPO smoke test.

## Smoke configuration

- Algorithm: MaskablePPO
- Safety shield: enabled
- Task-valid action masking: enabled
- Seed: 42
- Rollout updates: 3
- Environment timesteps: 3,072
- PPO optimization epochs: 30
- Parallel environments: 4

## Validation

| Metric | Result |
|---|---:|
| Tests | 59 passed |
| Invalid masked actions | 0 |
| Inspect collapse detected | False |
| Rollout safety cost | 0.0000 |
| Feasible validation safety cost | 0.0000 |
| Feasible validation hazard recall | 0.2467 |
| Feasible validation success | 0.0000 |
| Mean shield interventions | 1.4000 |
| Shield intervention rate | 0.0056 |

## Interpretation

The shield successfully prevents worker-proximity, restricted-zone, and
collision violations under the current oracle semantic map.

The refined action projection does not use hazard locations, reward values,
inspection progress, or hazard distance. It therefore acts as a safety
controller rather than a hidden task planner.

The intervention rate is approximately 0.56 percent of evaluation steps,
showing that the shield preserves most policy actions while enforcing zero
observed semantic safety cost.

The smoke evaluation uses only five scenarios and three rollout updates.
Task performance must therefore be measured using the complete Stage 3E-A
experiment.