# Stage 3B: Safe Viewpoint Inspection

## Motivation

The Stage 3A occupancy-based planner achieved full hazard recall only by
entering hazard cells. The strict safety planner completed fewer than half of
the validation and training scenarios because some hazard cells were inside
worker-exclusion regions.

A physical inspection robot should observe a hazard from a safe viewpoint
instead of occupying the hazard location.

## Inspection model

The inspection action now detects all uninspected hazards that satisfy both
conditions:

1. Manhattan distance is less than or equal to the inspection radius.
2. No obstacle blocks the Bresenham grid line between the robot and hazard.

The default inspection radius is two grid cells.

## Backward-compatible baseline

The Stage 3A occupancy planners remain available with an inspection radius of
zero. This supports a direct ablation between:

- occupancy-based inspection
- viewpoint-based inspection
- safety-aware viewpoint inspection

## Safety-aware planning

The safe viewpoint planner excludes:

- obstacle cells
- restricted-zone cells
- worker cells
- cells directly adjacent to workers

A start cell inside an exclusion region may be exited, but it cannot be
selected as an inspection viewpoint.

## Research value

This abstraction prepares the project for camera-based perception. A future
detector can replace the oracle line-of-sight signal while preserving the same
risk-map and reinforcement-learning interfaces.