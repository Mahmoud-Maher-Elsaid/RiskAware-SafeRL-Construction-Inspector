# Stage 4B2 Live Webots Sensors

## Objective

Stage 4B2 connects the Stage 4B observation bridge to live Webots sensor
readings without connecting MaskablePPO to the robot motors.

## Live devices

The Stage 4B2 robot contains:

- GPS
- InertialUnit
- Compass
- eight proximity sensors
- Emitter
- Receiver

The Supervisor contains a matching Emitter and Receiver pair.

## Live data flow

~~~text
GPS and InertialUnit
    -> world pose
    -> grid position and cardinal heading

Proximity sensors
    -> normalized sensor snapshot

Semantic scene contract
    -> seven-channel observation
    -> four-value state vector

Observation bridge
    -> task-valid action mask

Robot Emitter
    -> telemetry packet
    -> Supervisor Receiver

Supervisor Emitter
    -> acknowledgement
    -> robot Receiver
~~~

## Telemetry contract

Each telemetry packet contains:

- simulation time
- world position
- yaw
- grid position
- cardinal heading
- eight proximity readings
- compass vector
- semantic map length
- semantic map nonzero count
- semantic map sum
- normalized state vector
- five-action task mask
- acknowledgement count

## Safety boundary

The Stage 4B2 controller uses a deterministic movement sequence only for
sensor validation.

The trained policy is not loaded and no raw policy action is sent to the
motors.

The mandatory deployment order remains:

~~~text
live Webots sensors
    -> observation bridge
    -> task-valid mask
    -> MaskablePPO proposal
    -> semantic safety shield
    -> deadlock-safe fallback
    -> motion execution
~~~

## Stage 4B2A scope

Stage 4B2A creates and tests:

- the telemetry schema
- live robot and Supervisor controllers
- a generated Stage 4B2 world
- pose and communication devices
- source-world isolation

Runtime execution and evidence validation are completed in Stage 4B2B.

## Webots coordinate system

The Stage 4B2 world explicitly uses the EUN coordinate system:

~~~text
X positive: East
Y positive: Up
Z positive: North
~~~

This matches the robot model, the ground plane, the GridFrame x-z mapping,
and the bridge heading convention.

The Stage 4A source world is not modified by the Stage 4B2 world builder.

