# Stage 4B Observation and Action Bridge

## Objective

Stage 4B converts Webots state into the observation and action contracts used
by the validated SafeRL policy.

## Observation contract

The policy receives:

- a flattened seven-channel semantic map
- a four-value normalized state vector

The semantic channels are:

1. obstacles
2. uninspected hazards
3. workers
4. restricted zones
5. visited cells
6. agent position
7. semantic risk

For a 12 by 12 environment, the flattened map contains 1008 values.

The state vector contains:

1. normalized agent row
2. normalized agent column
3. normalized episode progress
4. inspected-hazard ratio

## Action contract

The discrete policy actions are:

1. move up
2. move down
3. move left
4. move right
5. inspect

The action bridge converts absolute grid actions into relative differential
drive primitives.

Example:

~~~text
MOVE_UP while facing EAST
    -> TURN_LEFT
    -> MOVE_FORWARD
~~~

## Action mask

The Stage 4B task mask blocks:

- movement outside the grid
- movement into known obstacles
- inspection when no visible hazard is inspectable

Worker proximity and restricted-zone actions remain available at the task-mask
layer. They will be handled by the mandatory semantic safety shield.

## Safety boundary

Stage 4B1 does not connect MaskablePPO to the robot motors.

The required runtime pipeline remains:

~~~text
Webots sensors
    -> observation bridge
    -> task-valid action mask
    -> MaskablePPO proposal
    -> semantic safety shield
    -> motion plan
    -> wheel commands
~~~

The raw policy action must never be sent directly to the motors.

## Limitation

Stage 4B1 validates the bridge using a semantic scene contract.

GPS, Compass, InertialUnit, Emitter, Receiver, and live Webots state will be
connected in Stage 4B2.
