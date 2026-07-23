# PPE Dataset Integration

## Local dataset

The PPE dataset is stored locally inside the repository working directory:

`Personal Protective Equipment - Combined Model.v1i.yolov8`

The dataset directory is excluded from Git and must not be committed.

The optional environment variable is:

`RISK_AWARE_PPE_DATASET_ROOT`

When the environment variable is not set, the repository-relative dataset directory is used.

## Dataset source of truth

The dataset `data.yaml` file defines the authoritative class order.

The expected class order is:

1. Fall-Detected
2. Gloves
3. Goggles
4. Hardhat
5. Ladder
6. Mask
7. NO-Gloves
8. NO-Goggles
9. NO-Hardhat
10. NO-Mask
11. NO-Safety Vest
12. Person
13. Safety Cone
14. Safety Vest

The class order must not be sorted alphabetically or changed during training, export, inference, tracking, or Webots integration.

## Audit result

The complete structural and annotation audit passed.

The audit verified:

- Image and label parity
- Label row format
- Class ID validity
- Normalized coordinates
- Bounding-box area
- Bounding-box bounds
- Duplicate annotation rows
- Image readability
- Missing labels
- Orphan labels

The audit reports are:

- `data/manifests/ppe_dataset_audit.json`
- `data/manifests/ppe_class_counts.csv`

## Generated integration files

The dataset integration command creates:

- `configs/perception/ppe_dataset.yaml`
- `configs/perception/class_schema.json`
- `data/manifests/ppe_dataset_manifest.json`

## Training policy

Production training must use a new experiment directory.

Historical custom PPE checkpoints must not be resumed.

An official generic pretrained backbone may be used for initialization.

The selected production model must originate from a new project-specific training run.
