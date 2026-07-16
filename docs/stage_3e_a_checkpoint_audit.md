# Stage 3E-A Checkpoint and Shield Audit

## Decision

The safety-selected best checkpoint is the canonical Stage 3E-A deployment
model when combined with the semantic runtime shield.

The final checkpoint is retained as an aggressive task-learning comparator,
but it requires more shield interventions and has zero validation success when
shielded.

## Shielded evaluation

| Model | Evaluation | Scenarios | Reward | Safety cost | Recall | Coverage | Success | Mean interventions |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Best | Feasible | 153 | 2.9416 | 0.0000 | 0.2728 | 0.0780 | 0.0065 | 52.82 |
| Best | Full | 200 | 2.2608 | 0.0000 | 0.2539 | 0.0755 | 0.0050 | 58.06 |
| Final | Feasible | 153 | 2.9095 | 0.0000 | 0.2731 | 0.0802 | 0.0000 | 63.33 |
| Final | Full | 200 | 2.2340 | 0.0000 | 0.2553 | 0.0768 | 0.0000 | 70.78 |

The best checkpoint reduces mean shield interventions by approximately
16.6 percent on feasible validation and 18.0 percent on full validation while
preserving nearly the same hazard recall.

## Unshielded evaluation

| Model | Evaluation | Scenarios | Reward | Safety cost | Recall | Coverage | Success |
|---|---|---:|---:|---:|---:|---:|---:|
| Best | Feasible | 153 | -17.3827 | 39.4379 | 0.3993 | 0.1260 | 0.0458 |
| Best | Full | 200 | -18.7536 | 40.9500 | 0.3909 | 0.1267 | 0.0350 |
| Final | Feasible | 153 | -18.6624 | 43.3791 | 0.4436 | 0.1493 | 0.0588 |
| Final | Full | 200 | -21.3732 | 48.1000 | 0.4430 | 0.1505 | 0.0500 |

## Interpretation

The runtime shield satisfies the evaluation safety constraint and reduces all
observed shielded safety costs to zero.

The unshielded policies remain substantially above the safety-cost limit of
5.0. The policy trained behind the shield therefore relies on runtime action
replacement rather than learning an intrinsically safe behavior policy.

The final checkpoint is more aggressive: it achieves greater unshielded
hazard recall but also produces greater unshielded safety cost.

Stage 3E-B will optimize counterfactual proposed-action costs so that unsafe
policy proposals receive a learning penalty even when the runtime shield
replaces them before execution.

## Canonical artifacts

- Deployment baseline: best model with semantic safety shield
- Aggressive task comparator: final model without shield
- Intrinsically safe policy candidate: none