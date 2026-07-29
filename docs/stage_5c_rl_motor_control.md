# Stage 5C RL Motor Control

Stage 5C is the repository's dedicated bounded policy-motor runtime. It is separate from the deterministic Stage 5A3/5B baseline.

The verified control path is:

`Webots GPS/inertial state + semantic scene + live CV risk -> checkpoint-compatible observation -> task-valid mask -> MaskablePPO proposal -> RuntimeSafetyShield -> executed action -> differential-drive primitives -> Webots wheel motors`

The selected checkpoint is the deadlock-safe shield MaskablePPO evaluation model with SHA-256 `172437CAE45B69031F443C0707FB0795D2F1860D3B95594BE281645D8A173FE7`.

The PPE model runs on CUDA during each policy decision. Interpreted risk updates the semantic risk channel before inference. Raw detections never directly command wheel velocities.

The accepted runtime recorded 10 decisions, two distinct policy actions, three restricted-zone shield interventions, three distinct wheel-command pairs, 14 command changes, and 10 successful CUDA perception inferences. Manual and fallback control were both false.

Evidence is stored under `reports/final_project_completion/stages/stage-5c-rl-motor-control/evidence/`.
