# Stage 4C1B Runtime Evidence

## Validation status

Stage 4C1B live MaskablePPO inference dry run passed static, unit,
repository, Webots, telemetry, and policy-isolation validation.

## Runtime results

- Robot telemetry samples: 30
- Policy proposals: 30
- Matched live samples: 30
- First policy sample index: 0
- Last policy sample index: 29
- Invalid proposals: 0
- Unique proposed actions: [1,2,3,4]

## Live observation verification

- Semantic map verified: True
- State vector verified: True
- Task-valid action mask verified: True
- All action masks respected: True

## Motor isolation

- Robot controller unchanged: True
- Policy connected to motors: False
- Policy actions applied: False
- Motor isolation verified: True

## Verified runtime path

```text
live Webots sensors
    -> existing observation bridge
    -> live telemetry
    -> isolated policy sidecar
    -> observation reconstruction verification
    -> task-valid action mask
    -> MaskablePPO proposal
    -> structured proposal log
```

The following path remained disconnected:

```text
MaskablePPO proposal -X-> wheel motors
```

## Scope boundary

Stage 4C1B proves that the trained MaskablePPO checkpoint can process
observations reconstructed from live Webots telemetry while respecting
the task-valid action mask.

The policy proposal was never applied to the robot.

The semantic safety shield and deadlock-safe fallback are not yet part
of the live Webots execution path.

The required future execution order remains:

```text
live sensors
    -> observation bridge
    -> task-valid action mask
    -> MaskablePPO proposal
    -> semantic safety shield
    -> deadlock-safe fallback
    -> executed motion primitive
```

This evidence does not constitute an absolute safety guarantee.
