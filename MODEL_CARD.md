# Model Card

## Policies

| Model | Path | SHA-256 | Status |
|---|---|---|---|
| PPO | `artifacts/final_submission/ppo/best_model.zip` | `7c54226fc8927d7ef7fcdc7d5f3ff29b11fa8d1463293a9d8e3b6bebad84cc11` | 30,208 training steps; zero final mission success |
| SAC | `artifacts/final_submission/sac/best_model.zip` | `78fe38fcc78df0ba8f98bd07af373848833876e7ec5ddf1486cec9a3cd9ba44c` | 10,000 training steps; zero final mission success |
| RiskShield-PPO | `artifacts/final_submission/riskshield_ppo/best_model.zip` | `d6c91caeb9bb10db866cbfe947ef1cc3db0ea8dc7470d098c9af6298ab75c812` | 15,300 steps; constrained telemetry; zero final mission success |

The full hashes are stored in checkpoint metadata and the Stage 9 manifest.
Checkpoints are local large artifacts and are intentionally excluded from Git.

## Perception

The production detector declares 14 classes. Its SHA-256 is
`4bd2190a3c99ffa5d1a7f57a37683c908c044e91485cd01e44ca4e199bde8550`.
CUDA inference
was verified on an RTX 3070 Ti Laptop GPU. Unsupported hazard types use
explicit simulator truth, not detector claims.

## Intended use and limitations

Models are for reproducible simulation research. They must not be used to make
real workplace safety decisions. Distribution shift, limited detector classes,
weak learned-policy task performance, and model-based shield assumptions are
material risks.
