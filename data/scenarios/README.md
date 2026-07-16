# RiskAware Construction Scenario Dataset

This directory contains deterministic construction-site scenario splits.

## Splits

- `train`: 1,000 scenarios
- `validation`: 200 scenarios
- `test_seen`: 200 scenarios
- `test_unseen`: 200 higher-density scenarios
- `stress`: 300 safety-critical scenarios

## Validation

The generator verifies:

- valid coordinates
- no overlapping entities
- reachable hazards
- unique layouts across splits
- deterministic generation
- SHA256 file integrity

## Regeneration

```powershell
.\.venv\Scripts\python.exe scripts\generate_scenarios.py `
    --output-dir data\scenarios `
    --seed 20260716 `
    --overwrite
```