from __future__ import annotations

from dataclasses import dataclass


@dataclass
class LagrangeMultiplier:
    """Projected gradient-ascent multiplier for a cost constraint."""

    value: float = 0.0
    learning_rate: float = 0.01
    maximum: float = 100.0

    def __post_init__(self) -> None:
        if self.value < 0.0:
            raise ValueError("value must be non-negative.")
        if self.learning_rate <= 0.0:
            raise ValueError("learning_rate must be positive.")
        if self.maximum <= 0.0:
            raise ValueError("maximum must be positive.")
        if self.value > self.maximum:
            raise ValueError("value cannot exceed maximum.")

    def update(
        self,
        observed_cost: float,
        cost_limit: float,
    ) -> float:
        if observed_cost < 0.0:
            raise ValueError("observed_cost must be non-negative.")
        if cost_limit < 0.0:
            raise ValueError("cost_limit must be non-negative.")

        self.value += self.learning_rate * (observed_cost - cost_limit)
        self.value = min(
            self.maximum,
            max(0.0, self.value),
        )
        return self.value

    def penalized_reward(
        self,
        reward: float,
        cost: float,
    ) -> float:
        if cost < 0.0:
            raise ValueError("cost must be non-negative.")
        return reward - self.value * cost
