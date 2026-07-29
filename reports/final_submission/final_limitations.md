# Final Limitations

- PPO, SAC, and RiskShield-PPO achieved zero complete-mission success.
- RiskShield-PPO's Lagrangian multiplier saturated at its configured cap.
- Evidence is simulation-only; Stage 5C is a bounded integrated runtime.
- The detector supports only 14 recorded classes. Unsupported concepts use
  explicitly labeled simulator truth.
- No trained recurrent checkpoint exists, so no recurrent comparison is made.
- Hardware validation covers Windows, Webots R2025a, and one RTX 3070 Ti Laptop
  GPU configuration.
- The predictive model and semantic map can be wrong under distribution shift.
- No certification or real-world safety guarantee is provided.
