# Stage 5B Live Perception Integration

## Scope

Stage 5B connects a real CUDA PPE detector to frames produced by the
Webots inspection camera.

This stage does not enable policy motor control. The deterministic
closed-loop waypoint controller remains the motor source until a
separate bounded policy-control runtime is verified.

## Selected model

The deployment candidate is selected with a semantic-first policy:

1. The candidate must load and execute successfully.
2. Mean inference latency must not exceed 20 milliseconds on the
   Stage 5B benchmark machine.
3. Important detected class diversity is prioritized.
4. Total detections, violation coverage, confidence, and latency are
   used as secondary criteria.

This corrects the initial latency-heavy score that could assign a
non-trivial score to a model producing zero detections.

The authoritative artifact path and SHA256 are stored in:

`configs/perception/stage5b_live_perception.json`

## Runtime contract

The backend:

- Accepts BGR or Webots BGRA frames.
- Verifies the configured model SHA256.
- Requires the expected 14-class PPE schema.
- Requires CUDA when configured.
- Produces typed detections.
- Produces a bounded semantic risk summary.
- Records inference latency.
- Can generate annotated evidence frames.
- Reports model connectivity truthfully.

## Safety boundaries

A missing detection is not evidence that a scene is safe.

Perception results do not control motors in Stage 5B2.

The runtime does not claim collision avoidance, real-world safety, or
complete PPE compliance.

## Verification

The Stage 5B2 validator replays the real Stage 5A3 Webots evidence
frames through the selected CUDA model and verifies:

- Model loading and SHA256.
- CUDA inference.
- One inference result per frame.
- Typed detection serialization.
- Risk aggregation.
- Annotated evidence generation.
- Truthful `cv_model_connected` reporting.

Live in-loop Webots integration is the next Stage 5B runtime gate.