from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from stable_baselines3.common.callbacks import BaseCallback

CURRICULUM_STAGE_INDEX = {
    "easy": 0,
    "medium": 1,
    "full": 2,
}


def curriculum_stage_for_update(
    completed_updates: int,
    *,
    easy_updates: int,
    medium_updates: int,
) -> str:
    if completed_updates < 0:
        raise ValueError("completed_updates must be non-negative.")
    if easy_updates < 1:
        raise ValueError("easy_updates must be positive.")
    if medium_updates < 1:
        raise ValueError("medium_updates must be positive.")

    if completed_updates < easy_updates:
        return "easy"
    if completed_updates < easy_updates + medium_updates:
        return "medium"
    return "full"


class CurriculumProgressCallback(BaseCallback):
    def __init__(
        self,
        output_path: str | Path,
        *,
        easy_updates: int,
        medium_updates: int,
        initial_completed_updates: int = 0,
        verbose: int = 0,
    ) -> None:
        super().__init__(verbose=verbose)
        curriculum_stage_for_update(
            initial_completed_updates,
            easy_updates=easy_updates,
            medium_updates=medium_updates,
        )
        self.output_path = Path(output_path)
        self.easy_updates = easy_updates
        self.medium_updates = medium_updates
        self.completed_updates = initial_completed_updates
        self.current_stage: str | None = None

    def _init_callback(self) -> None:
        self.output_path.parent.mkdir(parents=True, exist_ok=True)

    def _write_record(self, record: dict[str, Any]) -> None:
        with self.output_path.open("a", encoding="utf-8") as file:
            file.write(json.dumps(record) + "\n")

    def _set_stage(self, stage: str) -> None:
        if stage == self.current_stage:
            return

        states = self.training_env.env_method("set_curriculum_stage", stage)
        self.current_stage = stage
        active_count = int(states[0]["active_scenario_count"])
        self.logger.record(
            "curriculum/stage_index",
            CURRICULUM_STAGE_INDEX[stage],
        )
        self.logger.record(
            "curriculum/active_scenario_count",
            active_count,
        )
        self._write_record(
            {
                "event": "stage_transition",
                "completed_updates": self.completed_updates,
                "timesteps": int(self.num_timesteps),
                "stage": stage,
                "active_scenario_count": active_count,
            }
        )

        if self.verbose:
            print(
                "Curriculum stage changed: "
                f"stage={stage}, active_scenarios={active_count}, "
                f"completed_updates={self.completed_updates}"
            )

    def _on_training_start(self) -> None:
        stage = curriculum_stage_for_update(
            self.completed_updates,
            easy_updates=self.easy_updates,
            medium_updates=self.medium_updates,
        )
        self._set_stage(stage)

    def _on_rollout_start(self) -> None:
        stage = curriculum_stage_for_update(
            self.completed_updates,
            easy_updates=self.easy_updates,
            medium_updates=self.medium_updates,
        )
        self._set_stage(stage)

    def _on_step(self) -> bool:
        return True

    def _on_rollout_end(self) -> None:
        self.completed_updates += 1
        self._write_record(
            {
                "event": "rollout_end",
                "completed_updates": self.completed_updates,
                "timesteps": int(self.num_timesteps),
                "stage": self.current_stage,
            }
        )
