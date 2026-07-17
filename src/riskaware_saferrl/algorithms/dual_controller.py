from __future__ import annotations

from dataclasses import dataclass


@dataclass
class DualVariableController:
    """Slow, normalized, bounded dual-variable controller."""

    cost_limit: float
    learning_rate: float = 0.01
    maximum: float = 2.0
    ema_beta: float = 0.9
    warmup_updates: int = 5
    value: float = 0.0
    ema_cost: float | None = None
    update_count: int = 0

    def __post_init__(self) -> None:
        if self.cost_limit < 0.0:
            raise ValueError("cost_limit must be non-negative.")
        if self.learning_rate <= 0.0:
            raise ValueError("learning_rate must be positive.")
        if self.maximum <= 0.0:
            raise ValueError("maximum must be positive.")
        if not 0.0 <= self.ema_beta < 1.0:
            raise ValueError("ema_beta must be in [0, 1).")
        if self.warmup_updates < 0:
            raise ValueError("warmup_updates must be non-negative.")
        if not 0.0 <= self.value <= self.maximum:
            raise ValueError("value must be within [0, maximum].")

    def update(self, observed_episode_cost: float) -> float:
        if observed_episode_cost < 0.0:
            raise ValueError("observed_episode_cost must be non-negative.")

        self.update_count += 1

        if self.ema_cost is None:
            self.ema_cost = float(observed_episode_cost)
        else:
            self.ema_cost = self.ema_beta * self.ema_cost + (1.0 - self.ema_beta) * float(
                observed_episode_cost
            )

        if self.update_count <= self.warmup_updates:
            return self.value

        scale = max(self.cost_limit, 1.0)
        normalized_violation = (self.ema_cost - self.cost_limit) / scale

        self.value = min(
            self.maximum,
            max(
                0.0,
                self.value + self.learning_rate * normalized_violation,
            ),
        )
        return self.value

    def diagnostics(self) -> dict[str, float | int | None]:
        return {
            "value": float(self.value),
            "ema_cost": (None if self.ema_cost is None else float(self.ema_cost)),
            "cost_limit": float(self.cost_limit),
            "learning_rate": float(self.learning_rate),
            "maximum": float(self.maximum),
            "ema_beta": float(self.ema_beta),
            "warmup_updates": int(self.warmup_updates),
            "update_count": int(self.update_count),
        }
