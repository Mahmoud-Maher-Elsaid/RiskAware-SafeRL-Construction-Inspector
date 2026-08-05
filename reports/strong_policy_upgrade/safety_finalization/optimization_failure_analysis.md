# HRMPPO-Safe v3 optimization failure analysis

## Decision

The production replacement gate remains failed. RiskShield-PPO v1 remains the
production model. No CV replacement, Webots v3 acceptance, Benchmark v2,
paper-result update, release merge, or main-branch merge is authorized.

## Verified repairs

- Decomposed all 5,460 legacy constraint increments.
- Added event-aware hard, soft, controlled, raw, success-conditioned, and
  progress-normalized Safety Contract v3 metrics.
- Separated hard human contact from soft one-cell near-miss exposure while
  preserving the legacy counter.
- Added five independent vector cost critics and anti-windup PID multipliers.
- Fixed soft-budget units from raw episode totals to cost per 100 steps.
- Fixed shield transition credit so PPO uses the executed action that caused
  reward and cost, rather than the rejected proposal.
- Added persistent observed-worker memory and confidence-calibrated clearance
  confirmation under perception false negatives.
- Generated 51,000 genuine policy-visited safety corrections. The accepted
  repair used contiguous recurrent context and KL task anchoring.

## Strategy results

| Candidate | Steps | Success | Hardest | Recall | Legacy reduction | Raw cost | Result |
|---|---:|---:|---:|---:|---:|---:|---|
| v2 baseline | 100,000 | 76.67% | 70.00% | 94.25% | 0.00% | 35.64 | unsafe baseline |
| v3 strategy A | 5,000 | 25.56% | 0.00% | 63.54% | 82.01% | 14.50 | inactivity-biased failure |
| v3 strategy B | 5,000 | 23.33% | 0.00% | 70.33% | 84.78% | 15.11 | task collapse |
| v3 strategy C | 5,000 | 25.56% | 0.00% | 68.74% | 86.67% | 15.11 | task collapse |
| strategy B | 30,000 | 52.22% | 0.00% | 85.38% | 48.10% | 26.72 | gate failed |
| strategy B | 100,000 | 45.56% | 0.00% | 82.68% | 51.34% | 26.32 | multiplier saturation/task regression |
| corrected units | 30,000 | 45.56% | 20.00% | 80.51% | 45.71% | 27.76 | gate failed |
| executed-action credit | 30,000 | 46.67% | 10.00% | 80.76% | 46.37% | 26.39 | gate failed |

## Remaining causal blocker

The paired event traces contain hard events whose shield decision is
`accept`: the corresponding worker or restricted cell was not present in the
causal observation before entry. At 20% perception false-negative probability
and partial observation, the required rates (restricted entry at most 0.005
per episode and human-clearance breach at most 0.01 per episode) effectively
permit no event in a 90-episode evaluation. The tested observation-only
confidence confirmation reduced diagnostic contact events but did not make
the full paired protocol pass without mission collapse.

This does not prove that no future algorithm can improve the trade-off. It
does prove that none of the completed candidates satisfies the unchanged
contract, so replacement and downstream acceptance are prohibited.
