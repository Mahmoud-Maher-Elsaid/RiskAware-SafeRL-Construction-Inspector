from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback

ACTION_NAMES = (
    "move_up",
    "move_down",
    "move_left",
    "move_right",
    "inspect",
)


class ActionDiagnosticsCallback(BaseCallback):
    """Track action collapse, safety costs, and terminal episode metrics."""

    def __init__(
        self,
        output_path: str | Path,
        *,
        verbose: int = 0,
    ) -> None:
        super().__init__(verbose=verbose)

        self.output_path = Path(output_path)
        self.action_counts = np.zeros(len(ACTION_NAMES), dtype=np.int64)
        self.transition_count = 0
        self.cost_sums = {
            "safety_cost": 0.0,
            "collision_cost": 0.0,
            "worker_cost": 0.0,
            "restricted_cost": 0.0,
        }
        self.terminal_records: list[dict[str, float]] = []

    def _init_callback(self) -> None:
        self.output_path.parent.mkdir(parents=True, exist_ok=True)

    def _on_rollout_start(self) -> None:
        self.action_counts.fill(0)
        self.transition_count = 0

        for key in self.cost_sums:
            self.cost_sums[key] = 0.0

        self.terminal_records.clear()

    def _on_step(self) -> bool:
        actions = np.asarray(self.locals.get("actions", []), dtype=np.int64).reshape(-1)

        if actions.size:
            self.action_counts += np.bincount(
                actions,
                minlength=len(ACTION_NAMES),
            )[: len(ACTION_NAMES)]

        infos = self.locals.get("infos", [])
        dones = np.asarray(self.locals.get("dones", np.zeros(len(infos), dtype=bool))).reshape(-1)

        for index, info in enumerate(infos):
            self.transition_count += 1
            self.cost_sums["safety_cost"] += float(info.get("cost", 0.0))
            self.cost_sums["collision_cost"] += float(info.get("cost_collision", 0.0))
            self.cost_sums["worker_cost"] += float(info.get("cost_worker", 0.0))
            self.cost_sums["restricted_cost"] += float(info.get("cost_restricted", 0.0))

            if index < dones.size and bool(dones[index]):
                self.terminal_records.append(
                    {
                        "hazard_recall": float(info.get("hazard_recall", 0.0)),
                        "coverage": float(info.get("coverage", 0.0)),
                        "success": float(bool(info.get("success", False))),
                    }
                )

        return True

    def _on_rollout_end(self) -> None:
        total_actions = int(np.sum(self.action_counts))
        transition_denominator = max(1, self.transition_count)

        record: dict[str, Any] = {
            "timesteps": int(self.num_timesteps),
            "total_actions": total_actions,
            "transitions": self.transition_count,
            "action_counts": {},
            "action_frequencies": {},
            "mean_step_costs": {},
            "completed_episodes": len(self.terminal_records),
        }

        for action_index, action_name in enumerate(ACTION_NAMES):
            count = int(self.action_counts[action_index])
            frequency = count / max(1, total_actions)

            record["action_counts"][action_name] = count
            record["action_frequencies"][action_name] = frequency

            self.logger.record(
                f"diagnostics/action_frequency_{action_name}",
                frequency,
            )

        for cost_name, cost_sum in self.cost_sums.items():
            mean_cost = cost_sum / transition_denominator
            record["mean_step_costs"][cost_name] = mean_cost
            self.logger.record(
                f"diagnostics/mean_step_{cost_name}",
                mean_cost,
            )

        if self.terminal_records:
            for metric_name in ("hazard_recall", "coverage", "success"):
                mean_value = float(
                    np.mean(
                        [terminal_record[metric_name] for terminal_record in self.terminal_records]
                    )
                )

                record[f"episode_{metric_name}_mean"] = mean_value
                self.logger.record(
                    f"diagnostics/episode_{metric_name}_mean",
                    mean_value,
                )

        with self.output_path.open("a", encoding="utf-8") as file:
            file.write(json.dumps(record) + "\n")
