from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from riskaware_saferrl.envs import ResearchConstructionEnvV2
from riskaware_saferrl.safety.safety_contract_v3 import (
    ControlledInspectionState,
    SafetyContractV3,
    SafetyTransitionRecord,
)


@dataclass(frozen=True)
class SafetyStepContext:
    proposed_action: int
    predicted_trajectory: tuple[tuple[int, int], ...]
    predicted_vector_cost: Any
    shield_result: dict[str, Any]
    inspection: ControlledInspectionState


class SafetyContractV3Wrapper(gym.Wrapper):
    """Create one typed, executed-action-aligned safety record per transition."""

    def __init__(
        self,
        environment: ResearchConstructionEnvV2,
        contract: SafetyContractV3,
    ) -> None:
        super().__init__(environment)
        self.contract = contract
        self.context: SafetyStepContext | None = None
        self.last_transition: SafetyTransitionRecord | None = None
        self.episode_id = "uninitialized"
        self._finished = False
        original = environment.observation_space
        self.observation_space = spaces.Dict(
            {
                "map": original["map"],
                "state": spaces.Box(-1.0, 1.0, (22,), dtype=np.float32),
                "action_mask": original["action_mask"],
            }
        )
        self._current_observation: dict[str, np.ndarray] | None = None

    @staticmethod
    def augment_observation(
        observation: dict[str, np.ndarray],
        vector_cost: tuple[float, ...],
    ) -> dict[str, np.ndarray]:
        if len(vector_cost) != 5:
            raise ValueError("Safety Contract v3 observation requires five vector costs")
        return {
            "map": observation["map"],
            "state": np.concatenate(
                (
                    observation["state"].astype(np.float32, copy=False),
                    np.asarray(vector_cost, dtype=np.float32),
                )
            ),
            "action_mask": observation["action_mask"],
        }

    @property
    def research_environment(self) -> ResearchConstructionEnvV2:
        return self.env.unwrapped

    def reset(self, **kwargs):
        observation, info = self.env.reset(**kwargs)
        self.contract.reset()
        self.context = None
        self.last_transition = None
        self._finished = False
        seed = kwargs.get("seed")
        self.episode_id = f"safety-v3:{seed}"
        self._current_observation = self.augment_observation(observation, (0.0, 0.0, 0.0, 0.0, 0.0))
        return self._current_observation, info

    def prepare_step(self, context: SafetyStepContext) -> None:
        if self._finished:
            raise RuntimeError("Cannot prepare post-termination safety accounting")
        self.context = context

    def step(self, action: int):
        if self._finished:
            raise RuntimeError("Safety Contract v3 forbids post-termination accounting")
        if self.context is None:
            raise RuntimeError("Safety step context must be prepared before execution")
        context = self.context
        if self._current_observation is None:
            raise RuntimeError("Safety Contract v3 environment must be reset before step")
        observation_before = self._current_observation
        pre_collisions = self.research_environment.collisions
        pre_restricted = self.research_environment.restricted_violations
        pre_workers = set(self.research_environment.workers)
        next_observation, reward, terminated, truncated, info = self.env.step(action)
        environment = self.research_environment
        position = environment.agent
        collision = environment.collisions > pre_collisions
        restricted = environment.restricted_violations > pre_restricted
        human_clearance_breach = any(
            abs(position[0] - row) + abs(position[1] - column) <= 1 for row, column in pre_workers
        )
        semantic_risk = float(position in environment.ppe_risk)
        uncertainty = float(
            environment.config.perception_false_negative_rate
            * max(
                next_observation["map"][1].max(),
                next_observation["map"][2].max(),
                next_observation["map"][7].max(),
                next_observation["map"][8].max(),
            )
        )
        actual_vector = self.contract.vector_cost(
            collision=collision,
            restricted_zone=restricted,
            human_clearance_breach=human_clearance_breach,
            semantic_risk=semantic_risk,
            uncertainty=uncertainty,
            inspection=context.inspection,
        )
        controlled = self.contract.controlled_inspection_valid(context.inspection)
        augmented_next_observation = self.augment_observation(
            next_observation, actual_vector.as_tuple()
        )
        active = {}
        if actual_vector.collision:
            active["collision"] = ("hard", actual_vector.collision, False)
        if actual_vector.restricted_zone:
            active["restricted_zone_entry"] = (
                "hard",
                actual_vector.restricted_zone,
                False,
            )
        if actual_vector.worker_near_miss:
            active["human_clearance_breach"] = (
                "hard",
                actual_vector.worker_near_miss,
                False,
            )
        if semantic_risk:
            active[
                "controlled_inspection_exposure" if controlled else "uncontrolled_semantic_risk"
            ] = (
                "controlled" if controlled else "soft",
                0.05 if controlled else semantic_risk,
                controlled,
            )
        if actual_vector.uncertainty:
            active["uncertainty_exposure"] = (
                "soft",
                actual_vector.uncertainty,
                False,
            )
        shield_decision = str(context.shield_result.get("shield_decision", "unknown"))
        event_updates = self.contract.update_events(
            episode_id=self.episode_id,
            step=environment.steps,
            position=position,
            active=active,
            proposed_action=context.proposed_action,
            executed_action=int(action),
            shield_decision=shield_decision,
            inspection_intent=context.inspection.inspection_intent,
        )
        legacy_constraint_delta = int(collision) + int(restricted) + int(human_clearance_breach)
        self.contract.add_legacy(
            raw_cost=float(info.get("cost", 0.0)),
            constraint_count=legacy_constraint_delta,
        )
        if terminated or truncated:
            event_updates = (
                *event_updates,
                *self.contract.finish_episode(
                    success=bool(info["success"]),
                    mission_progress=float(info["inspection_coverage"]),
                    resolution_action=int(action),
                ),
            )
            self._finished = True
        self.last_transition = SafetyTransitionRecord(
            observation=observation_before,
            policy_action=context.proposed_action,
            action_mask=tuple(bool(value) for value in observation_before["action_mask"]),
            predicted_trajectory=context.predicted_trajectory,
            predicted_vector_cost=context.predicted_vector_cost,
            shield_result=context.shield_result,
            executed_action=int(action),
            next_state=augmented_next_observation,
            actual_vector_cost=actual_vector,
            event_updates=tuple(event_updates),
            terminated=bool(terminated),
            truncated=bool(truncated),
            legacy_raw_safety_cost=float(info.get("cost", 0.0)),
        )
        enriched = dict(info)
        enriched["safety_vector_v3"] = actual_vector.as_dict()
        enriched["safety_contract_v3"] = self.contract.summary()
        enriched["safety_transition_v3"] = self.last_transition
        self._current_observation = augmented_next_observation
        self.context = None
        return augmented_next_observation, reward, terminated, truncated, enriched
