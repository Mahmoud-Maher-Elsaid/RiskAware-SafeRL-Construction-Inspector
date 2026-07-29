# PPE Detector Model Card

- Artifact: `artifacts/runs/perception_production_100e/yolo26s_100e_seed42_20260722_214549/weights/best.pt`
- SHA-256: `4bd2190a3c99ffa5d1a7f57a37683c908c044e91485cd01e44ca4e199bde8550`
- Framework: Ultralytics YOLO, PyTorch
- Production device: CUDA (`cuda:0`)
- Input size: 640 px
- Confidence threshold: 0.25

## Trained classes

Fall-Detected, Gloves, Goggles, Hardhat, Ladder, Mask, NO-Gloves,
NO-Goggles, NO-Hardhat, NO-Mask, NO-Safety Vest, Person, Safety Cone, and
Safety Vest.

The model does **not** claim visual detection of holes, arbitrary machinery,
restricted signs, scaffolding state, or every construction hazard. Those layers
come from explicitly labeled simulator ground truth or configuration during
simulation experiments.

The selected checkpoint had seven detections across three important classes in
the Stage 5B selection benchmark, mean inference latency 12.08 ms, and a small
evaluation sample. These figures do not establish broad real-world accuracy.
Absence of a detection is never treated as proof of safety.
