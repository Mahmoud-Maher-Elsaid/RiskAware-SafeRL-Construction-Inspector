# Stage 3A Expert Baseline Results

## Validation benchmark

The complete validation split contains 200 fixed scenarios.

| Baseline | Mean reward | Mean safety cost | Hazard recall | Success rate |
|---|---:|---:|---:|---:|
| Inspect only | -38.125 | 21.250 | 0.000 | 0.000 |
| Random | -62.533 | 78.385 | 0.184 | 0.000 |
| Greedy A* | 20.709 | 5.950 | 1.000 | 1.000 |
| Safe Greedy A* | 17.724 | 0.020 | 0.824 | 0.395 |

## Expert-plan feasibility

The safe planner completed 377 of 1,000 training scenarios.

- complete plans: 377
- incomplete plans: 623
- completion rate: 0.377

## Interpretation

The unconstrained oracle confirms that the scenarios are geometrically
solvable. However, the strict safety planner cannot inspect every hazard in
most scenarios without entering worker-exclusion or restricted cells.

The current inspection action requires the robot to occupy the hazard cell.
This is not an appropriate final abstraction for physical construction
inspection because a robot should inspect many hazards from a safe viewpoint.

## Decision

Long PPO training is postponed until Stage 3B introduces safe inspection
viewpoints and validates constrained scenario feasibility.
