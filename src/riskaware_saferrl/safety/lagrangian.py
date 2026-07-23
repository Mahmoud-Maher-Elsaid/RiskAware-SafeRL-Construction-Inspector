from dataclasses import dataclass


@dataclass
class LagrangeMultiplier:
    """Projected gradient-ascent multiplier for a safety-cost constraint."""

    value: float = 0.0
    learning_rate: float = 0.01
    maximum: float = 100.0

    def update(self, observed_cost: float, cost_limit: float) -> float:
        self.value += self.learning_rate * (observed_cost - cost_limit)
        self.value = min(self.maximum, max(0.0, self.value))
        return self.value

    def penalized_reward(self, reward: float, cost: float) -> float:
        return reward - self.value * cost
