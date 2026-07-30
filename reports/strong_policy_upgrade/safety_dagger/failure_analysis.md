# Targeted safety DAgger failure analysis

## Result

The first 51,000-correction run passed dataset integrity but failed task
competence. Final held-out success was 16.67%, and hazard recall and inspection
coverage were 40.08%. The checkpoint is rejected.

## Root causes

1. Only selected risk states were written, but records retained their original
   episode identifier. The recurrent sequence loader therefore interpreted
   non-contiguous selected states as adjacent timesteps. This invalidated the
   temporal training contract.
2. Risk-aware A* disagreed with the successful policy on 70.63% of iteration-1
   visits. Offline cross-entropy did not provide the required explicit KL task
   anchor.
3. Deadlock selection was level-triggered. It contributed 11,008 records in
   iteration 2 and overwhelmed mission-progress examples.
4. A 20% safe-example probability applied only when no risk reason existed.
   Persistent uncertainty and semantic-risk predictions therefore suppressed
   most task anchors.

## Repair

The immutable failed chunks and checkpoints remain preserved. The repair run
stores contiguous policy-visited trajectories, retains risk reasons as
metadata, uses the successful systematic causal expert with the predictive
shield, and keeps the original causal dataset in every cumulative retraining
stage. No failed result is relabeled.
