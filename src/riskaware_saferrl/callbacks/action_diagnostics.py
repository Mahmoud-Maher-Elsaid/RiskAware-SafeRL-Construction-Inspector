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
    def __init__(
        self,
        output_path: str | Path,
        *,
        collapse_threshold: float = 0.95,
        collapse_patience: int = 3,
        verbose: int = 0,
    ) -> None:
        super().__init__(verbose=verbose)

        if not 0.0 < collapse_threshold <= 1.0:
            raise ValueError("collapse_threshold must be in (0, 1].")

        if collapse_patience < 1:
            raise ValueError("collapse_patience must be positive.")

        self.output_path = Path(output_path)
        self.collapse_threshold = collapse_threshold
        self.collapse_patience = collapse_patience
        self.action_counts = np.zeros(len(ACTION_NAMES), dtype=np.int64)
        self.transition_count = 0
        self.cost_sums = {
            "safety_cost": 0.0,
            "collision_cost": 0.0,
            "worker_cost": 0.0,
            "restricted_cost": 0.0,
        }
        self.terminal_records: list[dict[str, float]] = []
        self.mask_transition_count = 0
        self.valid_action_sum = 0
        self.inspect_available_count = 0
        self.invalid_masked_action_count = 0
        self.collapse_streak = 0

    def _init_callback(self) -> None:
        self.output_path.parent.mkdir(parents=True, exist_ok=True)

    def _on_rollout_start(self) -> None:
        self.action_counts.fill(0)
        self.transition_count = 0

        for key in self.cost_sums:
            self.cost_sums[key] = 0.0

        self.terminal_records.clear()
        self.mask_transition_count = 0
        self.valid_action_sum = 0
        self.inspect_available_count = 0
        self.invalid_masked_action_count = 0

    def _record_masks(
        self,
        actions: np.ndarray,
    ) -> None:
        raw_masks = self.locals.get("action_masks")

        if raw_masks is None:
            return

        masks = np.asarray(raw_masks, dtype=np.bool_)

        if masks.ndim == 1:
            masks = masks.reshape(1, -1)

        if masks.ndim != 2 or masks.shape[1] != len(ACTION_NAMES):
            return

        record_count = min(actions.size, masks.shape[0])

        if record_count == 0:
            return

        selected_actions = actions[:record_count]
        selected_masks = masks[:record_count]
        row_indices = np.arange(record_count)

        valid_selected_actions = selected_masks[
            row_indices,
            selected_actions,
        ]

        self.mask_transition_count += record_count
        self.valid_action_sum += int(np.count_nonzero(selected_masks))
        self.inspect_available_count += int(np.count_nonzero(selected_masks[:, 4]))
        self.invalid_masked_action_count += int(np.count_nonzero(~valid_selected_actions))

    def _on_step(self) -> bool:
        actions = np.asarray(
            self.locals.get("actions", []),
            dtype=np.int64,
        ).reshape(-1)

        if actions.size:
            self.action_counts += np.bincount(
                actions,
                minlength=len(ACTION_NAMES),
            )[: len(ACTION_NAMES)]

        self._record_masks(actions)

        infos = self.locals.get("infos", [])
        dones = np.asarray(
            self.locals.get(
                "dones",
                np.zeros(len(infos), dtype=bool),
            )
        ).reshape(-1)

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

        inspect_frequency = float(record["action_frequencies"]["inspect"])

        if inspect_frequency >= self.collapse_threshold:
            self.collapse_streak += 1
        else:
            self.collapse_streak = 0

        collapse_detected = self.collapse_streak >= self.collapse_patience

        record["collapse_diagnostics"] = {
            "threshold": self.collapse_threshold,
            "patience": self.collapse_patience,
            "streak": self.collapse_streak,
            "detected": collapse_detected,
        }

        self.logger.record(
            "diagnostics/inspect_collapse_streak",
            self.collapse_streak,
        )
        self.logger.record(
            "diagnostics/inspect_collapse_detected",
            float(collapse_detected),
        )

        if collapse_detected and self.verbose:
            print(
                "Inspect-action collapse warning: "
                f"frequency={inspect_frequency:.6f}, "
                f"streak={self.collapse_streak}"
            )

        for cost_name, cost_sum in self.cost_sums.items():
            mean_cost = cost_sum / transition_denominator
            record["mean_step_costs"][cost_name] = mean_cost
            self.logger.record(
                f"diagnostics/mean_step_{cost_name}",
                mean_cost,
            )

        if self.mask_transition_count:
            mask_denominator = max(1, self.mask_transition_count)
            mask_metrics = {
                "transitions": self.mask_transition_count,
                "mean_valid_action_count": (self.valid_action_sum / mask_denominator),
                "inspect_available_frequency": (self.inspect_available_count / mask_denominator),
                "invalid_action_count": (self.invalid_masked_action_count),
                "invalid_action_rate": (self.invalid_masked_action_count / mask_denominator),
            }
            record["action_mask_metrics"] = mask_metrics

            for metric_name, metric_value in mask_metrics.items():
                if metric_name == "transitions":
                    continue

                self.logger.record(
                    f"diagnostics/mask_{metric_name}",
                    metric_value,
                )

        if self.terminal_records:
            for metric_name in (
                "hazard_recall",
                "coverage",
                "success",
            ):
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
