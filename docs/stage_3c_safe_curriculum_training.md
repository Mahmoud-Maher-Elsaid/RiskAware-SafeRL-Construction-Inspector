# Stage 3C: Safe-Feasible Curriculum PPO Training

## Objective

Stage 3C converts the safe-viewpoint feasibility analysis into a deterministic
training curriculum and performs the first complete PPO learning run.

## Curriculum construction

A scenario is included in the safe-feasible curriculum only when the
safety-aware viewpoint planner:

- completes the inspection plan
- reaches full hazard recall
- finishes successfully
- accumulates zero safety cost

The selected inspection radius remains two grid cells.

## Difficulty score

Safe-feasible training scenarios are ranked using a documented heuristic that
combines:

- movement actions
- inspection actions
- obstacle count
- worker count
- restricted-zone count

The ranked scenarios are divided into balanced easy, medium, and hard thirds.
The source scenario-manifest hash is stored in the curriculum manifest so that
training cannot silently use a different dataset version.

## PPO schedule

The default 100-update run uses:

- updates 1-25: easy tier
- updates 26-50: easy and medium tiers
- updates 51-100: all safe-feasible tiers

With four environments and 256 steps per environment, this produces 102,400
environment timesteps. Ten PPO optimization epochs per rollout produce 1,000
optimization epochs in total.

## Evaluation

Periodic deterministic model selection uses a fixed safe-feasible validation
subset. Final evaluation reports both:

- all safe-feasible validation scenarios
- the complete validation split

The complete validation split prevents curriculum filtering from hiding model
failures on scenarios that remain difficult under the strict safety model.

## Generated outputs

Training outputs remain outside version control under `artifacts/runs` and
`artifacts/tensorboard`. The deterministic curriculum manifest is stored under
`configs/curriculum` as a small reproducibility artifact.
