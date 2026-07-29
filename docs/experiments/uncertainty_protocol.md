# Perception Uncertainty Protocol

The protocol crosses false-negative rates 0%, 10%, 20%, and 30% with normal,
low-light, motion-blur, and camera-noise observations. It also evaluates low,
medium, and high moving-obstacle density plus a held-out layout configuration.
Five deterministic evaluation seeds yield 320 episodes.

Three data layers remain separate:

1. clean simulator ground truth used only for metrics and the predictive shield;
2. deterministically perturbed semantic perception;
3. the flattened observation delivered to the policy.

False negatives drop only active perceptual cells. Low light scales confidence
and adds deterministic dropout; motion blur applies a horizontal three-cell
kernel; camera noise adds seeded Gaussian noise. No ground-truth label is
modified.

For condition \(c\), robustness is

\[
R_c = \mathrm{clip}_{[0,1]}\left[
\frac{1}{2}\operatorname{mean}
\left(\frac{\mathrm{recall}_c}{\mathrm{recall}_0},
\frac{\mathrm{coverage}_c}{\mathrm{coverage}_0},
\frac{\mathrm{success}_c}{\mathrm{success}_0}\right)
+\frac{1}{2}\operatorname{mean}_{m\in S}
\frac{1}{1+\max(0,m_c-m_0)}
\right],
\]

where \(S\) contains collision, near-miss, and constraint-violation rates. If
baseline success is zero, the success ratio is neutral (one) and the negative
result is reported separately.
