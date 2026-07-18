# Stage 4B2 Runtime Evidence

## Validation status

Stage 4B2 live Webots sensor integration passed static and runtime
validation.

## Runtime results

- Robot telemetry samples: 30
- Supervisor telemetry samples: 30
- Successfully transferred samples: 30
- Maximum acknowledgement count: 29
- First sample index: 0
- Last sample index: 29
- Unique grid positions: 3
- Unique headings: 2
- Observed headings: EAST, SOUTH
- World-position span: 0.9109649816714055

## Observation contract

- Flattened semantic map length: 1008
- State-vector length: 4
- Task action-mask length: 5
- Proximity sensor count: 8
- Compass output validated: True
- Action mask validated: True
- Runtime verified: True

## Observed grid positions

```text
[
  [
    5,
    5
  ],
  [
    6,
    5
  ],
  [
    6,
    6
  ]
]
```

## Coordinate convention

The Stage 4B2 Webots world explicitly uses the EUN coordinate system:

```text
X positive: East
Y positive: Up
Z positive: North
```

This convention matches the robot geometry, the ground plane, the GPS
x-z mapping, and the GridFrame contract.

## Scope boundary

This validation demonstrates live sensor acquisition, observation
construction, task-valid action masking, telemetry transfer, and
acknowledgements inside Webots.

The trained MaskablePPO policy is not connected to the motors in
Stage 4B2.

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
