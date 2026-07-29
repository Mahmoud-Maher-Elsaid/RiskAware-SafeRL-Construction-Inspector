# Stage 4A Runtime Validation

## Runtime robot

Stage 4A uses the built-in Webots e-puck differential-drive robot to validate
the integration pipeline before connecting the trained policy.

The runtime test validates:

- left and right wheel velocity control
- wheel encoder readings
- eight proximity sensor readings
- forward motion
- left rotation
- right rotation
- safe stop
- Supervisor pose reset
- Supervisor physics reset
- controller log creation

## Controller roles

`manual_robot_test` executes a deterministic motion sequence and records
sensor data.

`scenario_supervisor` restores the robot pose, resets its physics state,
verifies the reset, and terminates the simulation with a process status.

`construction_robot` is a fail-safe placeholder. It keeps both wheel
velocities at zero until the policy, observation adapter, and semantic shield
are connected in later stages.

## Acceptance evidence

A successful run creates:

~~~text
webots/logs/manual_robot_test.jsonl
webots/logs/scenario_supervisor.jsonl
~~~

The manual controller must report at least one valid sensor sample.

The Supervisor must record:

~~~json
{"event": "reset_verification", "reset_ok": true}
~~~

The runtime logs are local artifacts and are excluded from Git.

## Self-contained robot model

The Stage 4A world uses a project-local Robot node instead of a vendor
EXTERNPROTO.

The robot contains:

- two HingeJoint wheel assemblies
- two RotationalMotor devices
- two PositionSensor devices
- eight DistanceSensor devices
- a rigid body and passive caster contact
- the existing deterministic test controller

This removes the runtime dependency on an optional vendor robot package and
keeps Stage 4A reproducible across Webots installations.
