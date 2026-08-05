# CV Final Audit

The active live-perception configuration is `configs/webots/live_perception.json`. It selects the 14-class Ultralytics detector at `artifacts/runs/perception_production_100e/yolo26s_100e_seed42_20260722_214549/weights/best.pt` with SHA-256 `4bd2190a3c99ffa5d1a7f57a37683c908c044e91485cd01e44ca4e199bde8550`, CUDA device `cuda:0`, 640-pixel inference, confidence threshold 0.25, and fail-closed loading.

The available Stage 5B3 runtime evidence verifies CUDA inference, class-schema compatibility, live detections, and temporal cadence. It reports mean inference latency 21.9243 ms and p95 latency 45.7501 ms in the validated runtime evidence (the separate model-selection benchmark reported 12.0822 ms mean and 18.4408 ms p95).

No complete independent labeled validation artifact sufficient to recompute mAP50, mAP50-95, or critical-class recall was found in the current repository state. Those metrics are therefore unavailable here; no fabricated replacement metric is reported. The production checkpoint is retained unchanged. The runtime explicitly records that absence of a detection is not a safety claim.
