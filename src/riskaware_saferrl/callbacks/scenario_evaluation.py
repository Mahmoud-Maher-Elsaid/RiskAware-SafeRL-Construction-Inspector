from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from stable_baselines3.common.callbacks import BaseCallback

from riskaware_saferrl.evaluation.scenario_evaluator import (
    compute_selection_score,
    evaluate_policy_on_scenarios,
    save_evaluation_results,
)
from riskaware_saferrl.scenarios import Scenario


class ScenarioEvaluationCallback(BaseCallback):
    def __init__(
        self,
        scenarios: Sequence[Scenario],
        *,
        eval_freq: int,
        output_directory: str | Path,
        selection_metric: str,
        safety_cost_limit: float,
        use_shield: bool = False,
        use_action_masks: bool = False,
        evaluate_at_start: bool = True,
        verbose: int = 1,
    ) -> None:
        super().__init__(verbose=verbose)

        if eval_freq < 1:
            raise ValueError("eval_freq must be positive.")

        if not scenarios:
            raise ValueError("At least one validation scenario is required.")

        self.scenarios = tuple(scenarios)
        self.eval_freq = eval_freq
        self.output_directory = Path(output_directory)
        self.selection_metric = selection_metric
        self.safety_cost_limit = safety_cost_limit
        self.use_shield = use_shield
        self.use_action_masks = use_action_masks
        self.evaluate_at_start = evaluate_at_start
        self.best_score = float("-inf")

    def _init_callback(self) -> None:
        self.output_directory.mkdir(parents=True, exist_ok=True)
        (self.output_directory / "best_model").mkdir(
            parents=True,
            exist_ok=True,
        )

    def _on_training_start(self) -> None:
        if self.evaluate_at_start:
            self._run_evaluation()

    def _on_step(self) -> bool:
        if self.n_calls % self.eval_freq == 0:
            self._run_evaluation()

        return True

    def _run_evaluation(self) -> None:
        records, summary = evaluate_policy_on_scenarios(
            self.model,
            self.scenarios,
            use_shield=self.use_shield,
            use_action_masks=self.use_action_masks,
            deterministic=True,
        )

        score = compute_selection_score(
            summary,
            selection_metric=self.selection_metric,
            safety_cost_limit=self.safety_cost_limit,
        )

        summary["timesteps"] = int(self.num_timesteps)
        summary["selection_metric"] = self.selection_metric
        summary["selection_score"] = score
        summary["safety_cost_limit"] = self.safety_cost_limit

        output_name = f"step_{self.num_timesteps:012d}"

        save_evaluation_results(
            records,
            summary,
            self.output_directory,
            output_name,
        )

        history_path = self.output_directory / "evaluation_history.jsonl"

        with history_path.open("a", encoding="utf-8") as file:
            file.write(json.dumps(summary) + "\n")

        metrics: dict[str, Any] = summary["metrics"]

        for metric_name in (
            "reward",
            "safety_cost",
            "collision_cost",
            "worker_cost",
            "restricted_cost",
            "hazard_recall",
            "coverage",
            "success",
        ):
            self.logger.record(
                f"validation/{metric_name}_mean",
                float(metrics[metric_name]["mean"]),
            )

        self.logger.record(
            "validation/selection_score",
            score,
        )

        if score > self.best_score:
            self.best_score = score
            best_model_path = self.output_directory / "best_model" / "best_model"
            self.model.save(best_model_path)

            best_summary_path = self.output_directory / "best_model" / "best_summary.json"
            best_summary_path.write_text(
                json.dumps(summary, indent=2) + "\n",
                encoding="utf-8",
            )

            if self.verbose:
                print(
                    "New best validation model saved: "
                    f"score={score:.6f}, "
                    f"timesteps={self.num_timesteps}"
                )
