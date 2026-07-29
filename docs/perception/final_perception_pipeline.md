# Final Perception and Semantic Risk Pipeline

The production flow is:

Webots camera → BGR preprocessing → Ultralytics YOLO on CUDA → typed
`LiveDetection` records → confidence/NMS filtering → temporal smoothing and
duplicate suppression → robot-local projection → semantic risk map → observation
and action-mask update → policy proposal → Safety Shield.

The detector performs framework NMS at inference and uses a 0.25 confidence
threshold. The runtime verifies the model SHA-256 and CUDA device, records
latency and failures, fails closed after repeated errors, and saves deterministic
annotations. `TemporalDetectionFilter` smooths confidence, suppresses
same-cell/class duplicates, and expires stale tracks. `SemanticRiskMapper`
projects the bounding-box bottom point using the calibrated horizontal FOV into
a robot-local grid.

Semantic sources are explicit:

- `cv`: only the 14 trained detector classes listed in the model card;
- `simulator_ground_truth`: holes, machinery state, and restricted-zone geometry
  used only in simulation benchmark channels;
- `configuration`: static inspection targets and route metadata.

A PPE violation or nearby person changes the CV risk map and can change the
action mask or emergency-stop state before action execution. Raw detections never
produce wheel commands directly.
