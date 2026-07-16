# Stage 3E Semantic Safety Shield

## Purpose

Stage 3D solved invalid-action and inspect-action collapse with task-valid
MaskablePPO. It did not satisfy the semantic safety-cost constraint.

Stage 3E-A introduces a deterministic semantic safety shield for worker
proximity and restricted zones.

## Separation of concerns

Task masks continue to remove only task-invalid actions:

- movement outside the grid
- movement into an obstacle
- inspection without an inspectable hazard

The semantic shield handles safety-validity separately:

- worker-proximity violations
- restricted-zone violations
- collision violations as a defensive fallback

## Replacement policy

A safe proposed action is executed unchanged.

An unsafe proposed movement is projected onto a safe task-valid action with
the smallest directional deviation from the proposed movement. Stable
action-index ordering resolves equal-deviation ties. Inspection is used only
when no safe task-valid movement is available.

The projection does not use hazard positions, hazard distance, inspection
progress, or reward information. This prevents the shield from acting as a
hidden task planner.

The shield records the proposed action, executed action, violation categories,
and replacement validity in the step information.

## Scientific interpretation

This shield is an action-replacement safety baseline. It can guarantee
one-step semantic safety under the current oracle map, but it changes the
action executed by the environment after the policy proposes an action.

It is not equivalent to constrained policy optimization. A separate adaptive
Lagrangian experiment will evaluate whether the policy itself can learn to
reduce semantic safety costs without relying entirely on runtime replacement.

## Deadlock handling

A semantic deadlock occurs when every task-valid action is unsafe.

The shield resolves these states in this order:

1. project onto a safe task-valid action
2. use the safe inspect action as an emergency hold when movement is unsafe
3. execute the least-unsafe task-valid action only when no safe action exists

Emergency holds are not exposed to the policy action mask. They are internal
runtime interventions. Unavoidable violations are explicitly recorded rather
than causing training to crash.

Inspection is considered unsafe when the current position already violates
worker-proximity or restricted-zone constraints.
