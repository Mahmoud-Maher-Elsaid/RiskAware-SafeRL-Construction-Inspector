# Stage 5A Live RGB Camera Acquisition

## Purpose

Stage 5A establishes a validated live RGB acquisition pipeline between
the Webots inspection camera and the future computer-vision subsystem.

This stage does not perform PPE detection or reinforcement-learning
motor control.

## Runtime pipeline

```text
Webots RGB camera
    -> copied BGRA frame buffer
    -> NumPy image
    -> image-quality metrics
    -> synchronized GPS, IMU, and compass telemetry
    -> JSONL evidence
    -> independent runtime validator
```

## Camera contract

The Stage 5A camera uses:

- device name: `inspection camera`
- resolution: `640 x 360`
- pixel format: `BGRA`
- expected bytes per frame: `921600`
- capture count: `120`
- capture stride: `4` simulation steps

## Image-quality evidence

Each captured frame records:

- SHA-256 checksum
- mean brightness
- contrast standard deviation
- grayscale entropy
- Laplacian sharpness
- non-black pixel ratio
- saturated pixel ratio
- temporal frame difference
- simulation timestamp
- motion phase
- GPS position
- roll, pitch, and yaw
- compass vector

Three lossless PNG frames are preserved as visual evidence.

## Control boundary

The robot follows a scripted validation motion sequence.

The following components remain disabled:

- live PPE inference
- worker tracking
- hazard classification
- MaskablePPO motor execution
- semantic-shield motor execution

Stage 5A proves that live visual and pose telemetry are valid before
adding a computer-vision model.

The fresh 2026-07-21 runtime captured 120 frames, 107 unique checksums, and 1.665 m of measured robot motion. The independent evidence is `webots/logs/stage5a_live_camera/stage5a_validation_report.json`. The modular perception configuration remains disabled because no production model artifact was found.

## Success criteria

Stage 5A succeeds only when:

- exactly 120 camera frames are captured
- every frame has the expected BGRA byte length
- timestamps are strictly increasing
- image brightness and contrast pass minimum thresholds
- entropy and sharpness pass minimum thresholds
- frames contain sufficient visible pixels
- frames change during robot motion
- at least 20 unique frame checksums are observed
- GPS, IMU, and compass telemetry are finite
- robot path length exceeds the validation threshold
- all three PNG evidence frames exist
- CV inference remains disabled
- policy motor control remains disabled
