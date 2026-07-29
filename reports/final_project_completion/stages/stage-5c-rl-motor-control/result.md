# Stage 5C RL Motor-Control Result

Status: **PASSED**

The dedicated Webots runtime loads the selected MaskablePPO checkpoint on CUDA, validates its 1008-element semantic map, four-element state vector, and five-action space, and runs deterministic masked inference.

Each policy proposal passes through the runtime SafetyShield. The executed action is converted into differential-drive primitives and the resulting signed wheel velocities are sent to the Webots left and right motors. The accepted run recorded 10 policy decisions, two distinct policy actions, three restricted-zone shield interventions, three distinct wheel-command pairs, and 14 motor-command changes.

The production PPE model ran live on CUDA for all 10 policy decisions with zero failures and 10 annotated frames. Interpreted CV risk changed the checkpoint-compatible semantic observation in nine decisions before policy inference. Raw detections never directly commanded the motors.

The run used no manual control and no fallback controller. The curated evidence contains the full proposal/shield/action trace, wheel-command log, runtime summary, annotated CV frame, and first-person viewport images.
