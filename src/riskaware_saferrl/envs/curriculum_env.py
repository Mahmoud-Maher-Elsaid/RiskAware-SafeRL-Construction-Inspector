from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from riskaware_saferrl.curriculum import CURRICULUM_TIERS
from riskaware_saferrl.envs.construction_env import ConstructionInspectionEnv
from riskaware_saferrl.scenarios import Scenario

CURRICULUM_STAGE_TIERS = {
    "easy": ("easy",),
    "medium": ("easy", "medium"),
    "full": CURRICULUM_TIERS,
}


class CurriculumConstructionInspectionEnv(ConstructionInspectionEnv):
    def __init__(
        self,
        scenario_tiers: Mapping[str, Sequence[Scenario]],
        *,
        inspection_radius: int = 2,
    ) -> None:
        missing_tiers = set(CURRICULUM_TIERS) - set(scenario_tiers)
        if missing_tiers:
            raise ValueError("Missing curriculum tiers: " + ", ".join(sorted(missing_tiers)))

        self._tier_scenarios = {tier: tuple(scenario_tiers[tier]) for tier in CURRICULUM_TIERS}

        if any(not scenarios for scenarios in self._tier_scenarios.values()):
            raise ValueError("Every curriculum tier must contain scenarios.")

        all_scenarios = tuple(
            scenario for tier in CURRICULUM_TIERS for scenario in self._tier_scenarios[tier]
        )
        scenario_ids = [scenario.scenario_id for scenario in all_scenarios]

        if len(scenario_ids) != len(set(scenario_ids)):
            raise ValueError("Curriculum scenarios must have unique identifiers.")

        self._scenario_tiers = {
            scenario.scenario_id: tier
            for tier, scenarios in self._tier_scenarios.items()
            for scenario in scenarios
        }
        self.curriculum_stage = "easy"
        self._active_scenarios = self._tier_scenarios["easy"]

        super().__init__(
            scenarios=all_scenarios,
            inspection_radius=inspection_radius,
        )

    @property
    def active_scenario_ids(self) -> tuple[str, ...]:
        return tuple(scenario.scenario_id for scenario in self._active_scenarios)

    def set_curriculum_stage(self, stage: str) -> dict[str, Any]:
        if stage not in CURRICULUM_STAGE_TIERS:
            raise ValueError(f"Unknown curriculum stage: {stage}")

        active_tiers = CURRICULUM_STAGE_TIERS[stage]
        self.curriculum_stage = stage
        self._active_scenarios = tuple(
            scenario for tier in active_tiers for scenario in self._tier_scenarios[tier]
        )
        return self.get_curriculum_state()

    def get_curriculum_state(self) -> dict[str, Any]:
        return {
            "stage": self.curriculum_stage,
            "active_tiers": list(CURRICULUM_STAGE_TIERS[self.curriculum_stage]),
            "active_scenario_count": len(self._active_scenarios),
        }

    def _select_scenario(
        self,
        options: dict[str, Any] | None,
    ) -> Scenario | None:
        if options is not None and "scenario_index" in options:
            return super()._select_scenario(options)

        scenario_index = int(self.np_random.integers(0, len(self._active_scenarios)))
        return self._active_scenarios[scenario_index]

    def _add_curriculum_info(self, info: dict[str, Any]) -> dict[str, Any]:
        enriched = dict(info)
        enriched["curriculum_stage"] = self.curriculum_stage
        enriched["curriculum_active_scenarios"] = len(self._active_scenarios)
        enriched["curriculum_scenario_tier"] = (
            self._scenario_tiers.get(self.current_scenario.scenario_id)
            if self.current_scenario is not None
            else None
        )
        return enriched

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        observation, info = super().reset(seed=seed, options=options)
        return observation, self._add_curriculum_info(info)

    def step(
        self,
        action: int,
    ) -> tuple[dict[str, np.ndarray], float, bool, bool, dict[str, Any]]:
        observation, reward, terminated, truncated, info = super().step(action)
        return (
            observation,
            reward,
            terminated,
            truncated,
            self._add_curriculum_info(info),
        )
