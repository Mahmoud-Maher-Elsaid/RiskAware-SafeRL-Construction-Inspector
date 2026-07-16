from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback

from riskaware_saferrl.safety.lagrangian import (
    LagrangeMultiplier,
)


class LagrangianUpdateCallback(BaseCallback):
    """Update a shared multiplier from completed episode costs."""

    def __init__(
        self,
        multiplier: LagrangeMultiplier,
        *,
        cost_limit: float,
        output_path: str | Path,
        verbose: int = 0,
    ) -> None:
        super().__init__(verbose=verbose)

        if cost_limit < 0.0:
            raise ValueError("cost_limit must be non-negative.")

        self.multiplier = multiplier
        self.cost_limit = cost_limit
        self.output_path = Path(output_path)
        self.step_cost_sum = 0.0
        self.transition_count = 0
        self.completed_episode_costs: list[float] = []

    def _init_callback(self) -> None:
        self.output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

    def _on_rollout_start(self) -> None:
        self.step_cost_sum = 0.0
        self.transition_count = 0
        self.completed_episode_costs.clear()

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        dones = np.asarray(
            self.locals.get(
                "dones",
                np.zeros(len(infos), dtype=bool),
            )
        ).reshape(-1)

        for index, info in enumerate(infos):
            self.transition_count += 1
            self.step_cost_sum += float(
                info.get(
                    "proposed_action_cost",
                    0.0,
                )
            )

            if index < dones.size and bool(dones[index]) and "episode_proposed_action_cost" in info:
                self.completed_episode_costs.append(float(info["episode_proposed_action_cost"]))

        return True

    def _on_rollout_end(self) -> None:
        mean_step_cost = self.step_cost_sum / max(1, self.transition_count)
        multiplier_before = float(self.multiplier.value)

        if self.completed_episode_costs:
            observed_episode_cost = float(np.mean(self.completed_episode_costs))
            multiplier_after = self.multiplier.update(
                observed_episode_cost,
                self.cost_limit,
            )
            updated = True
        else:
            observed_episode_cost = None
            multiplier_after = multiplier_before
            updated = False

        record: dict[str, Any] = {
            "timesteps": int(self.num_timesteps),
            "transitions": self.transition_count,
            "completed_episodes": len(self.completed_episode_costs),
            "mean_step_proposed_cost": (mean_step_cost),
            "mean_episode_proposed_cost": (observed_episode_cost),
            "cost_limit": self.cost_limit,
            "multiplier_before": multiplier_before,
            "multiplier_after": float(multiplier_after),
            "updated": updated,
        }

        self.logger.record(
            "lagrangian/mean_step_proposed_cost",
            mean_step_cost,
        )
        self.logger.record(
            "lagrangian/multiplier",
            float(multiplier_after),
        )

        if observed_episode_cost is not None:
            self.logger.record(
                "lagrangian/mean_episode_proposed_cost",
                observed_episode_cost,
            )

        with self.output_path.open(
            "a",
            encoding="utf-8",
        ) as file:
            file.write(json.dumps(record) + "\n")

        if self.verbose:
            print(
                "Lagrangian update: "
                f"episode_cost={observed_episode_cost}, "
                f"limit={self.cost_limit:.6f}, "
                f"lambda={multiplier_after:.6f}"
            )
