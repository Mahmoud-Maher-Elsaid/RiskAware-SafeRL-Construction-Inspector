# RiskShield-PPO

RiskShield-PPO is the project's constrained Safe RL method. It is not ordinary
PPO under a new label.

Let reward return be \(J_R(\theta)\), expected episodic safety cost be
\(J_C(\theta)\), and the cost budget be \(d\). Training uses the Lagrangian

\[
\max_\theta J_R(\theta) - \lambda(J_C(\theta)-d), \qquad \lambda \ge 0.
\]

At runtime each transition's reward presented to PPO is
\(r_t-\lambda c_t\). After an episode, projected gradient ascent updates
\(\lambda \leftarrow [\lambda + \eta(J_C-d)]_+\). Therefore observed cost and
the active multiplier directly modify PPO's policy objective.

An auxiliary PyTorch cost-value network estimates
\(V_C(s_t)\) with a one-step TD target
\(c_t+\gamma V_C(s_{t+1})\). Its MSE loss, the multiplier, raw and penalized
rewards, safety budget, episode cost, PPO losses, entropy, and KL statistics are
stored in training telemetry. The auxiliary critic is used for cost estimation
and diagnostics; PPO uses the multiplier-penalized reward for optimization.

The predictive shield is deliberately separable from constrained training.
Training defaults to shield-off so the policy observes the cost consequences of
unsafe proposals. Final evaluation compares PPO and RiskShield-PPO both with and
without the same k-step shield.

This method is validated in simulation only and does not establish real-world
construction safety.
