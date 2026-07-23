# YOLO26s 100-Epoch PPE Production Candidate

## Objective

This stage trains one strong YOLO26s production candidate for 100 complete epochs.

## Model decision

YOLO26s was promoted because the one-epoch pilot produced higher recall and higher mAP50 than YOLO26n while measured latency was nearly equal.

YOLO26n remains a lightweight deployment fallback.

## Dataset policy

Training uses:

- The complete train split
- The complete validation split

The test split is not included in the runtime data file and remains reserved for final unbiased evaluation.

## Initialization

Training starts from the locked official generic YOLO26s checkpoint.

The run does not use:

- Pilot checkpoints
- Historical custom PPE checkpoints
- Existing PPE-trained checkpoints
- Resumed optimizer state

## Training schedule

The run uses:

- 100 epochs
- Image size 640
- Automatic batch sizing targeting 70 percent GPU memory
- Automatic Mixed Precision
- Automatic optimizer selection
- Cosine learning-rate scheduling
- Multi-scale training
- Partial inverse-frequency class weighting
- Mosaic augmentation
- Mosaic shutdown during the final 15 epochs
- Full validation
- Checkpoint saving every 5 epochs

## Interruption handling

The active run path is recorded in:

`reports/perception/production_training/active_run.json`

The latest checkpoint is stored as `last.pt` inside the run directory.

Do not start a different training run after an interruption until the interrupted run has been inspected.

## Outputs

Local runtime dataset:

`artifacts/perception/production_100e`

Local training run:

`artifacts/runs/perception_production_100e`

Tracked reports:

`reports/perception/production_training`

## Acceptance

Completing 100 epochs creates a production candidate.

Final model acceptance requires:

- Full validation review
- Per-class safety metric review
- Reserved test-split evaluation
- Real-image qualitative inspection
- Webots live-camera inference
- Latency and stability validation
