from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from typing import Any

import gymnasium as gym

from riskaware_saferrl.envs import ResearchConstructionEnv

Position = tuple[int, int]


@dataclass(frozen=True)
class PredictiveShieldDecision:
    proposed_action: int
    predicted_trajectory: tuple[Position, ...]
    predicted_cost: float
    violation_types: tuple[str, ...]
    shield_decision: str
    replacement_action: int | None
    final_action: int
    confidence: float
    emergency_stop: bool
    computation_time_ms: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class PredictiveSafetyShield:
    """Deterministic k-step discrete safety projection."""

    def __init__(
        self,
        *,
        horizon: int = 3,
        safety_budget: float = 0.0,
        emergency_action: int = 4,
    ) -> None:
        if horizon < 1:
            raise ValueError("horizon must be positive")
        self.horizon = horizon
        self.safety_budget = safety_budget
        self.emergency_action = emergency_action

    def _predict(
        self, environment: ResearchConstructionEnv, action: int
    ) -> tuple[tuple[Position, ...], float, tuple[str, ...]]:
        position = environment.agent
        trajectory = [position]
        violations: list[str] = []
        total_cost = 0.0
        for _ in range(self.horizon):
            if action in environment.ACTION_TO_DELTA:
                delta = environment.ACTION_TO_DELTA[action]
                candidate = position[0] + delta[0], position[1] + delta[1]
                if not environment._inside(candidate) or candidate in environment.obstacles:  # noqa: SLF001
                    violations.append("collision")
                    total_cost += 1.0
                    trajectory.append(position)
                    break
                position = candidate
            trajectory.append(position)
            if position in environment.restricted:
                violations.append("restricted_zone")
                total_cost += 1.0
            if environment._near(  # noqa: SLF001
                position, environment.workers | environment.dynamic_hazards
            ):
                violations.append("near_miss")
                total_cost += 1.0
            if position in environment.ppe_risk:
                violations.append("unsafe_proximity")
                total_cost += 0.5
        if total_cost > self.safety_budget:
            violations.append("risk_budget")
        return tuple(trajectory), total_cost, tuple(dict.fromkeys(violations))

    def decide(
        self, environment: ResearchConstructionEnv, proposed_action: int
    ) -> PredictiveShieldDecision:
        started = time.perf_counter()
        proposed_trajectory, proposed_cost, violations = self._predict(environment, proposed_action)
        final_action = proposed_action
        replacement: int | None = None
        decision = "accept"
        emergency_stop = False
        if violations:
            candidates = []
            for action in range(environment.action_space.n):
                trajectory, cost, candidate_violations = self._predict(environment, action)
                candidates.append(
                    (
                        cost,
                        len(candidate_violations),
                        abs(action - proposed_action),
                        action,
                        trajectory,
                        candidate_violations,
                    )
                )
            best = min(candidates)
            if best[0] <= self.safety_budget and not best[5]:
                final_action = best[3]
                replacement = final_action
                decision = "replace"
            else:
                final_action = self.emergency_action
                replacement = final_action
                decision = "stop"
                emergency_stop = True
        elapsed = (time.perf_counter() - started) * 1000.0
        confidence = max(0.0, min(1.0, 1.0 - proposed_cost / max(1.0, self.horizon)))
        return PredictiveShieldDecision(
            proposed_action=int(proposed_action),
            predicted_trajectory=proposed_trajectory,
            predicted_cost=float(proposed_cost),
            violation_types=violations,
            shield_decision=decision,
            replacement_action=replacement,
            final_action=int(final_action),
            confidence=float(confidence),
            emergency_stop=emergency_stop,
            computation_time_ms=float(elapsed),
        )


class PredictiveShieldWrapper(gym.Wrapper):
    """Apply a predictive shield before forwarding an action."""

    def __init__(self, environment: ResearchConstructionEnv, shield: PredictiveSafetyShield):
        super().__init__(environment)
        self.shield = shield
        self.last_decision: PredictiveShieldDecision | None = None

    def step(self, action: int):
        self.last_decision = self.shield.decide(self.unwrapped, int(action))
        observation, reward, terminated, truncated, info = self.env.step(
            self.last_decision.final_action
        )
        enriched = dict(info)
        enriched.update(
            {
                "shield_active": self.last_decision.shield_decision != "accept",
                "shield_decision": self.last_decision.to_dict(),
                "proposed_action": self.last_decision.proposed_action,
                "executed_action": self.last_decision.final_action,
                "shield_intervention": self.last_decision.shield_decision != "accept",
                "emergency_stop": self.last_decision.emergency_stop,
            }
        )
        return observation, reward, terminated, truncated, enriched
