# Stage 3E-B Counterfactual Lagrangian Smoke Results

## Status

The counterfactual Lagrangian training pipeline passed the complete test suite
and CUDA smoke experiment.

## Configuration

- Algorithm: MaskablePPO
- Runtime semantic shield: enabled
- Counterfactual Lagrangian reward: enabled
- Primary model-selection evaluation: unshielded
- Seed: 42
- Rollout updates: 3
- Environment timesteps: 3,072
- PPO optimization epochs: 30
- Proposed episode-cost limit: 5.0
- Lagrange initial value: 0.0
- Lagrange learning rate: 0.01
- Lagrange maximum: 100.0

## Validation

| Metric | Result |
|---|---:|
| Tests | 66 passed |
| Lagrangian rollout records | 3 |
| Last-rollout completed episodes | 4 |
| Last mean proposed episode cost | 23.25 |
| Proposed episode-cost limit | 5.00 |
| Final Lagrange multiplier | 0.5225 |
| Primary evaluation shield | false |
| Unshielded feasible safety cost | 149.40 |
| Unshielded feasible hazard recall | 0.2533 |
| Shielded feasible safety cost | 0.00 |
| Shielded feasible hazard recall | 0.2467 |

## Interpretation

The training wrapper correctly computes safety cost from the action originally
proposed by the policy before runtime shield replacement.

The shared Lagrange multiplier increases when mean proposed episode cost is
above the configured limit. The policy reward therefore receives a nonzero
counterfactual safety penalty even though the executed shielded action has zero
observed safety cost.

The primary evaluation is unshielded, so checkpoint selection measures
intrinsic policy safety. A separate shielded evaluation measures runtime
deployment safety.

The three-update smoke experiment validates the implementation only. Its
unshielded safety cost is not a final performance result.