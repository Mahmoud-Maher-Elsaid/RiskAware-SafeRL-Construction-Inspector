# Stage 5C RL Motor-Control Plan

1. Audit all compatible policy checkpoints and select the strongest verified artifact.
2. Validate the Dict observation space, five-action space, feature extractor, mask interface, and checkpoint hash.
3. Build a dedicated Webots runtime that loads MaskablePPO, creates live semantic observations, applies task masks, evaluates the SafetyShield, and sends the executed action to wheel motors.
4. Run the PPE detector on CUDA during every policy decision and inject interpreted risk into the checkpoint-compatible semantic risk channel.
5. Record policy proposals, shield decisions, executed actions, and actual wheel velocities.
6. Exercise restricted-zone projection and emergency-stop unit behavior.
7. Run Webots end to end and accept only a no-manual, no-fallback result.
