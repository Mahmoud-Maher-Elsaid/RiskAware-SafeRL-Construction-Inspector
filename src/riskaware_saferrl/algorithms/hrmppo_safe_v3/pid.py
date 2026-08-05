from __future__ import annotations

from dataclasses import dataclass


@dataclass
class AntiWindupPIDMultiplier:
    value: float = 0.0
    proportional_gain: float = 0.05
    integral_gain: float = 0.001
    derivative_gain: float = 0.01
    maximum: float = 20.0
    integral_limit: float = 100.0
    warmup_updates: int = 10
    integral: float = 0.0
    previous_error: float = 0.0
    updates: int = 0
    saturated: bool = False

    def update(self, observed: float, budget: float, scale: float = 1.0) -> float:
        self.updates += 1
        normalized_error = (observed - budget) / max(abs(scale), 1e-8)
        derivative = normalized_error - self.previous_error
        self.previous_error = normalized_error
        if self.updates <= self.warmup_updates:
            return self.value
        candidate_integral = max(
            -self.integral_limit,
            min(self.integral_limit, self.integral + normalized_error),
        )
        update = (
            self.proportional_gain * normalized_error
            + self.integral_gain * candidate_integral
            + self.derivative_gain * derivative
        )
        candidate = self.value + update
        projected = max(0.0, min(self.maximum, candidate))
        self.saturated = projected != candidate
        # Anti-windup: integrate only when projection did not reject the
        # direction that would push the multiplier farther into saturation.
        if (
            not self.saturated
            or (projected == self.maximum and normalized_error < 0)
            or (projected == 0.0 and normalized_error > 0)
        ):
            self.integral = candidate_integral
        self.value = projected
        return self.value

    def state_dict(self) -> dict[str, float | int | bool]:
        return {
            "value": self.value,
            "integral": self.integral,
            "previous_error": self.previous_error,
            "updates": self.updates,
            "saturated": self.saturated,
        }

    def load_state_dict(self, state: dict[str, float | int | bool]) -> None:
        self.value = float(state["value"])
        self.integral = float(state["integral"])
        self.previous_error = float(state["previous_error"])
        self.updates = int(state["updates"])
        self.saturated = bool(state["saturated"])
