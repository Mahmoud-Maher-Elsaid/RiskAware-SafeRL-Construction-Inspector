# Research Methodology

The research question is whether a constrained policy and predictive shield can
improve safety without hiding inspection performance failures. Reward and cost
are separate. Evaluation uses paired seeds across algorithms and reports
coverage, recall, success, collisions, violations, cost, energy, latency, and
shield interventions.

The primary analysis uses nonparametric repeated-measures tests because metrics
are bounded and zero-inflated. Ablations isolate the shield, CV risk injection,
action masking, constrained objective, and prediction horizon. Negative and
identical results are retained.
