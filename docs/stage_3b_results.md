# Stage 3B Safe Viewpoint Results

## Selected inspection radius

The project uses an inspection radius of two grid cells.

This value provides the best balance between safe scenario feasibility and
task difficulty. A radius of three improves oracle completion but also makes
the random baseline substantially stronger. A radius of one remains too
restrictive for safe inspection.

## Validation radius comparison

| Radius | Inspect recall | Random recall | Random success | Safe recall | Safe success | Safe reward | Safe cost |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.0342 | 0.3916 | 0.005 | 0.9000 | 0.715 | 20.6328 | 0.0 |
| 2 | 0.0764 | 0.5250 | 0.030 | 0.9137 | 0.765 | 20.9852 | 0.0 |
| 3 | 0.1137 | 0.6135 | 0.145 | 0.9365 | 0.840 | 21.7092 | 0.0 |

## Radius-two feasibility

| Split | Scenarios | Completion rate | Hazard recall | Mean cost | Maximum cost |
|---|---:|---:|---:|---:|---:|
| Train | 1000 | 0.751 | 0.8955 | 0.0 | 0.0 |
| Validation | 200 | 0.765 | 0.9137 | 0.0 | 0.0 |
| Test seen | 200 | 0.825 | 0.9242 | 0.0 | 0.0 |
| Test unseen | 200 | 0.255 | 0.6503 | 0.0 | 0.0 |
| Stress | 300 | 0.070 | 0.4089 | 0.0 | 0.0 |

## Decision

Initial PPO curriculum training will use the safe-feasible training scenarios.
The complete scenario distribution will remain available for constrained
training, out-of-distribution evaluation, and stress testing.

The low test-unseen and stress completion rates are retained as intentional
generalization challenges rather than being hidden by increasing the
inspection radius.