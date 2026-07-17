# Stage 4 Webots Digital-Twin Integration

## Objective

Transfer the validated shielded MaskablePPO control architecture from the
deterministic grid-world environment to a Webots construction-site simulation.

## Canonical control pipeline

~~~text
Webots sensors
    -> observation adapter
    -> task-valid action mask
    -> MaskablePPO policy
    -> semantic safety shield
    -> motion primitive executor
    -> robot motors
~~~

The raw policy action must never be sent directly to the robot motors.

## Stage 4A scope

Stage 4A provides:

- a deterministic Webots construction-site world
- a differential-drive inspection robot
- a Python robot controller
- a Python Supervisor controller
- deterministic episode reset
- basic sensor validation
- basic motion-primitive validation
- structured simulation logs

Stage 4A does not perform reinforcement-learning training.

The trained MaskablePPO checkpoint will not control the robot during the
initial foundation tests.

## Planned directory structure

~~~text
webots/
├── assets/
├── config/
├── controllers/
│   ├── construction_robot/
│   ├── manual_robot_test/
│   └── scenario_supervisor/
├── logs/
├── protos/
└── worlds/

src/riskaware_saferrl/webots/
tests/webots/
~~~

## Stage 4A components

### Construction-site world

The first world will include:

- a flat construction floor
- static walls
- construction obstacles
- worker objects
- restricted-zone markers
- hazard markers
- an inspection robot
- a Supervisor node

### Inspection robot

The initial robot will use differential-drive motion with:

- left wheel motor
- right wheel motor
- wheel position sensors
- inertial unit
- GPS
- compass
- front distance sensor
- left distance sensor
- right distance sensor
- optional camera

### Robot controller

The robot controller will:

- initialize motors and sensors
- receive high-level motion commands
- execute deterministic motion primitives
- report robot pose and sensor readings
- stop safely when commands are invalid
- write structured diagnostic records

### Supervisor controller

The Supervisor controller will:

- reset the robot pose
- reset robot velocity
- place workers and hazards
- configure restricted zones
- start and stop test episodes
- collect episode-level metrics
- verify deterministic reset behavior

### Manual test controller

The manual controller will validate:

- stop
- move forward
- rotate left
- rotate right
- sensor availability
- emergency stop behavior

## Planned motion primitives

~~~text
STOP
MOVE_FORWARD
TURN_LEFT
TURN_RIGHT
INSPECT
~~~

The motion primitives will later provide the bridge between discrete policy
actions and continuous wheel velocities.

## Stage 4B plan

Stage 4B will implement the observation and action bridge:

~~~text
Webots sensor state
    -> semantic grid representation
    -> existing policy observation dictionary
~~~

It will also map policy actions to validated Webots motion primitives.

## Stage 4C plan

Stage 4C will connect the Version 1 policy:

~~~text
Webots observation
    -> action mask
    -> MaskablePPO
    -> semantic shield
    -> motion primitive
    -> wheel commands
~~~

The semantic shield remains mandatory.

## Stage 4D plan

Stage 4D will provide automated evaluation through the Supervisor.

The evaluation will report:

- hazard recall
- coverage
- success rate
- worker safety cost
- restricted-zone safety cost
- collision cost
- total safety cost
- shield-intervention rate
- emergency-hold rate
- least-unsafe fallback rate

Shielded and unshielded results must remain separate.

## Stage 4E plan

Stage 4E will introduce controlled domain randomization:

- floor friction
- object position
- obstacle geometry
- worker position
- worker movement speed
- sensor noise
- lighting
- camera exposure
- robot wheel slip

## Stage 4A acceptance criteria

Stage 4A is complete when:

- the Webots world opens without errors
- the robot controller starts successfully
- the Supervisor controller starts successfully
- the robot moves forward
- the robot rotates left
- the robot rotates right
- all configured sensors return valid readings
- the Supervisor resets the robot pose deterministically
- the Supervisor resets robot velocity
- one complete test episode is logged
- emergency stop behavior is validated
- all Python unit tests remain successful

## Safety rules

The trained policy must not be connected to the motors until:

1. manual motion primitives are validated
2. sensor readings are validated
3. deterministic reset behavior is validated
4. invalid commands trigger a safe stop
5. the observation adapter is tested
6. the action adapter is tested
7. the semantic shield path is tested
8. intervention diagnostics are verified

No claim of real-world safety is made from Webots simulation results alone.
