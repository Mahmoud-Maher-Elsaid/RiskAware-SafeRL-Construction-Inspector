# Stage 5A3 Closed-Loop Inspection Mission

## Purpose

Stage 5A3 replaces the earlier time-based demonstration motion with a
closed-loop waypoint inspection mission.

The controller continuously compares the robot's measured GPS position
and compass heading against the current waypoint. Quaternion-derived tilt
is monitored independently. Motor commands are updated
after every Webots simulation step.

## Runtime architecture

```text
RGB camera + GPS + IMU + compass + optional distance sensors
    -> runtime forward-direction calibration
    -> runtime differential-turn calibration
    -> waypoint distance and heading calculation
    -> closed-loop motor command
    -> waypoint arrival validation
    -> controlled inspection scan
    -> zone evidence capture
    -> return-to-start validation
    -> independent evidence validator
```

## Mission route

The mission follows a conservative U-shaped route around the validated
safe walkway. It includes eight route entries, multiple inspection
points, four controlled scan locations, and a final return to the start
area.

The route is stored in:

```text
configs/webots/stage5a3_closed_loop_route.json
```

## Closed-loop behavior

The controller uses:

- GPS position feedback
- IMU yaw feedback
- automatic forward-heading calibration
- automatic differential-turn sign calibration
- distance-proportional forward speed
- heading-proportional differential steering
- arrival tolerance validation
- stagnation detection
- bounded recovery turns
- optional proximity-based interventions
- controlled left, right, and recentered camera scans

## Evidence

The runtime preserves:

- timestamped JSONL telemetry
- waypoint arrival records
- route completion metrics
- path length
- accumulated heading change
- return-to-start distance
- camera quality metrics
- per-zone PNG evidence frames
- recovery and proximity intervention counts

## Control boundary

Stage 5A3 uses a deterministic closed-loop waypoint controller.

It does not use:

- PPE inference
- worker tracking
- hazard classification
- MaskablePPO motor execution
- semantic-shield motor execution

The runtime deliberately does not claim collision-free operation because
no dedicated physical contact sensor is used as proof.

## Success criteria

Stage 5A3 succeeds only when:

- navigation calibration completes
- every waypoint is reached within tolerance
- the full route completes
- the robot returns to the start area
- route length exceeds the configured minimum
- heading variation proves meaningful turning and scanning
- camera frames remain valid and temporally diverse
- zone evidence images are saved
- GPS, IMU, and compass telemetry remain finite
- recovery count remains bounded
- CV inference remains disabled
- policy motor control remains disabled

## Validated navigation conventions

Stage 5A3 uses explicit coordinate and motor conventions:

- `YAW_TO_WORLD_SIGN = -1.0`
- `FORWARD_MOTOR_SIGN = -1.0`

Negative equal wheel velocities move the chassis toward the
forward-facing inspection camera. The Webots IMU yaw sign is mapped to
the mathematical heading convention used on the XZ navigation plane.

The stagnation watchdog runs only while the robot is aligned closely
enough to make positional progress. Turning in place is not treated as
navigation stagnation.

## Physical stability and view correction

The original two-wheel model used a sphere that was rigidly included in
the main robot bounding object. Lowering that sphere to the floor stopped
the chassis from tipping, but it also created a high-resistance third
contact that prevented differential steering.

The final model uses front and rear passive `BallJoint` ball supports. Each sphere has
its own `Solid`, bounding object, and physics mass, so it can rotate
freely in all directions while supporting the rear of the chassis. The
sphere touches the construction slab at the initial robot height and is
not duplicated in the main body bounding object.

The drive-wheel track is 0.72 m and the main body uses a lowered center of mass inside the wheel-and-caster
support footprint. The controller records both wheel encoder positions,
roll, and pitch. It stops the motors and fails the mission when the
measured tilt exceeds the validated limit.

The inspection camera is positioned ahead of its visible housing. The
main Webots `Viewpoint` follows the robot position without inheriting
robot orientation, preserving a stable external mission view.


## Final stability and camera correction

The drive wheels and rear caster use separate contact materials. The
caster has low symmetric friction, while the wheels retain traction.
The caster anchor is farther behind the axle and the center of mass is
lower inside the support footprint.

Wheel torque, navigation speed, and turn speed are limited. Motor
commands use a slew-rate limiter, and the robot turns in place until it
is aligned within five degrees. Short tilt transients are tolerated,
but sustained or hard tilt still stops the mission.

The first valid camera image is saved to
`webots/logs/stage5a3_closed_loop/live_camera_preview.png`.

## Runtime verification on 2026-07-21

Three consecutive runs with the same configuration completed all eight waypoints and returned to the start region. Each run recorded a 10.571 m path, 408.576 s simulated duration, maximum absolute roll 9.572 degrees, maximum absolute pitch 9.296 degrees, zero recoveries, 1,596 captures, and 1,595 unique checksums. The independent report is `webots/logs/stage5a3_closed_loop/stage5a3_validation_report.json`; repeatability copies are under `webots/logs/stage5a3_repeatability/`.

Visible launch evidence is stored in `webots/logs/stage5a3_repeatability/visible_launch_check/launcher_record.json`. It records the Stage 5A3 world filename in the Webots R2025a window title and both controller startup markers. Explicit first, middle, and final frames are stored in the run's `evidence_frames` directory.
