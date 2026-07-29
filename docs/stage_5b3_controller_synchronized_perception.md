# Stage 5B3 Controller-Synchronized Live Perception

## Scope

Stage 5B3 executes the selected CUDA PPE model while the Stage 5A3
Webots inspection mission is active.

The Stage 5A3 controller writes bounded evidence frames. A separate
CUDA sidecar detects each newly completed frame, writes typed JSONL
records, generates annotated images, and records latency statistics.

## Safety boundary

Perception does not control motors in Stage 5B3. The deterministic
closed-loop waypoint controller remains the motor source.

A missing detection is not evidence that the scene is safe. The stage
does not claim collision-free operation because verified ranging-sensor
coverage is not available.

## Runtime verification

The validator requires:

- A verified Stage 5A3 mission.
- Route completion and return to the start location.
- A connected PPE model with CUDA inference.
- Perception execution while the mission is active.
- Processing and annotation of every evidence frame.
- Zero sidecar failures.
- No policy motor control.