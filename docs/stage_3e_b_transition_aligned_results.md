# Stage 3E-B Transition-Aligned Scalar Lagrangian Results

## Configuration

- Algorithm: MaskablePPO
- Semantic shield during training: disabled
- Action masking: enabled
- Scalar Lagrangian reward wrapper: enabled
- Calibration updates: 20
- Environment timesteps: 20,480
- PPO optimization epochs: 200
- Seed: 42
- Episode safety-cost limit: 5.0
- Lagrange learning rate: 0.01

## Validation

| Metric | Result |
|---|---:|
| Tests | 71 passed |
| Final Lagrange multiplier | 7.8800 |
| Final proposed episode cost | 36.7500 |
| Unshielded feasible safety cost | 48.1176 |
| Unshielded feasible hazard recall | 0.2046 |
| Unshielded full safety cost | 46.8650 |
| Unshielded full hazard recall | 0.1991 |
| Shielded feasible safety cost | 0.0000 |
| Shielded feasible hazard recall | 0.1785 |
| Shielded full safety cost | 0.0000 |
| Shielded full hazard recall | 0.1577 |

## Findings

The transition-aligned implementation removes the action-transition mismatch.
The policy action, executed action, reward, and resulting next state remain
consistent during training.

The scalar Lagrangian reward formulation does not satisfy the safety constraint.
The final episode cost remains substantially above the configured limit.

The multiplier increases continuously while episode cost remains noisy. The
resulting reward scale becomes strongly negative, value-function loss grows,
and policy updates become very small.

This method is retained as a mechanically valid but scientifically unsuccessful
baseline.

## Decision

The scalar reward-penalty method will not be trained for 100 updates.

The next stage implements a dual-critic constrained PPO method with:

- a reward value function
- a cost value function
- separate reward and cost generalized advantage estimates
- a constrained policy surrogate objective
- normalized reward and cost advantages
- a slower and bounded dual update
- unshielded primary evaluation
- shielded deployment evaluation
