# PPE Detector Training

## Training readiness

The local PPE dataset integration passed validation.

The verified environment includes:

- Python 3.11
- CUDA-enabled PyTorch
- NVIDIA GPU execution
- A locked 14-class dataset schema
- A complete dataset audit
- A permanent dataset integration validator

## Training policy

Production model development uses fresh project-specific experiments.

Historical custom PPE checkpoints must not be resumed.

Official generic pretrained weights may be used as initialization.

The test split is not used for pilot training, hyperparameter selection, or model selection.

## Initial pilot

The initial bounded pilot compares:

- YOLO26n
- YOLO26s

Each candidate is trained for one epoch on a deterministic stratified subset.

The subset includes all 14 dataset classes.

The pilot measures:

- Training completion
- CUDA memory use
- Batch-size feasibility
- Validation metrics
- Native PyTorch latency
- Checkpoint creation
- Artifact hashes

Pilot metrics are not final model-selection results.

The purpose of the pilot is to validate the training pipeline and determine feasible settings for longer controlled experiments.

## Outputs

Pilot datasets:

`artifacts/perception/pilot_datasets`

Generic pretrained weights:

`artifacts/models/perception/generic_pretrained`

Pilot runs:

`artifacts/runs/perception_pilots`

Pilot reports:

`reports/perception/pilot_runs`

Large model and runtime artifacts are local and excluded from Git.
