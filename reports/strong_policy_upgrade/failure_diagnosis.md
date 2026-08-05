# Learned-Policy Failure Diagnosis

The exact environment is solvable. Across 100 seeds per site, the best expert success rate was 100% on small, medium, and dynamic sites, exceeding all gates.

## Evidence-backed causes

- **insufficient_training_budget**: PPO used 30,208 steps, SAC 10,000, and v1 stopped at 15,300; every learned benchmark episode truncated.
- **spatial_structure_destroyed**: A 10x16x16 semantic tensor plus state was flattened to 2,570 scalars for an MLP.
- **memory_missing**: The environment is partially observed, but v1 uses feed-forward MlpPolicy with no recurrent state.
- **mask_train_inference_mismatch**: The environment exposes masks, but standard PPO/SAC training does not consume them.
- **constraint_scale_failure**: The safety budget is 5 while v1 training episodes commonly cost 90-160; lambda reached its cap of 100.
- **objective_double_penalty**: Collision, near-miss, restricted, and PPE events reduce reward and separately add safety cost before Lagrangian penalization.
- **premature_early_stopping**: Patience is 50 episodes and selection uses reward-cost rather than mission success/recall/coverage composite.
- **planner_information_advantage**: Planner risk cost reads full restricted/worker/dynamic sets; learned policies receive partial noisy observations.
- **observation_noise_inconsistency**: Detection channels resample false negatives on each observation, while the risk channel exposes visible ground-truth risks.
- **all_or_nothing_checkpoint_selection**: Success requires all hazards, but early stopping does not prioritize success and v1 checkpoints are final rather than best held-out composite.

The environment should not be made artificially easy. The v2 work must preserve the exact completion condition while fixing observation persistence, structured encoding, reward/cost scaling, mask use, memory, curriculum, and checkpoint selection.
