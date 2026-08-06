# Legacy gate audit

Date: 2026-08-06  
Repository head: `63a6a411f1741ac7f0615a35fbf2f9539c9e47ec`

The current accepted release is `PROJECT_COMPLETED_WITH_PRODUCTION_V1_RETAINED_AND_V4_EXPERIMENTAL`.
The release gates for software, production v1 Webots, corrected experimental v4 Webots, the 1,620-row benchmark, CV schema/CUDA audit, ablations, paper, release package, final acceptance, integration, main, and exact-head CI are passed.

Legacy research gates were audited against their machine-readable evidence:

- Causal Behavior Cloning passed: mean held-out accuracy 0.91069, best accuracy 0.91393, minimum primary per-action recall 0.84134, and memory gate passed.
- Flat DAgger completed three genuine policy-visited iterations and passed its summary gate. Its final held-out success was 0.76667 with mean safety cost 37.475; it remains a research baseline, not production.
- RiskShield-HRMPPO v2 was not accepted as a replacement. The strongest preliminary candidate reached 0.76667 overall success and 0.96667 target success, but mean safety cost 35.63611 exceeded the declared replacement criterion. The rejection is preserved rather than relabeled.
- RiskShield-HRMPPO-Safe v3 remains rejected by its preserved paired evidence.
- Hierarchical DAgger correction datasets exist, but the beta-zero mission gate failed; those checkpoints remain experimental evidence only.
- Hierarchical v4 is accepted as an experimental runtime and is not installed as production.

No historical checkpoint, demonstration, benchmark row, or failed result was regenerated or deleted during this audit. The remaining scientific limitations are the rejected replacement candidates and unavailable independent labeled CV metrics; neither invalidates the accepted v1-production/v4-experimental release.
