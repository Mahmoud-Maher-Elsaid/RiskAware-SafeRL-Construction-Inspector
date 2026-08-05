# Reward and Safety Cost v2

RiskShield-HRMPPO v2 uses the original five-action task and exact mission
completion condition, but separates task utility from constraint cost.
Historical v1 environments and benchmark rows remain unchanged.

The task reward is

\[
r_t=-c_{\mathrm{step}}+r_{\mathrm{explore}}+r_{\mathrm{inspect}}
+\alpha(\gamma\Phi(s_{t+1})-\Phi(s_t))-p_{\mathrm{inefficiency}}
+r_{\mathrm{terminal}}.
\]

The potential is the normalized negative Manhattan distance to the nearest
uninspected hazard. Its discounted difference encourages progress without
making a closed loop profitable. Inspection progress and explored-area
coverage are reported separately.

Safety events are not subtracted from task reward. They form the independent
cost

\[
c_t=1.0I_{\mathrm{collision}}+1.0I_{\mathrm{restricted}}
+0.5I_{\mathrm{near\ miss}}+0.25I_{\mathrm{PPE\ risk}}.
\]

The episodic safety budget is 20 cost units. This scale is deliberately above
any single avoidable event and is calibrated for the 180--400 step scenarios;
the constrained optimizer, rather than duplicate reward penalties, enforces
the budget.

V2 observations preserve detected static semantic evidence across an episode,
keep current dynamic entities separate, expose the action mask as a typed
field, encode orientation cyclically, and include previous action, reward, and
cost for recurrent inference.
