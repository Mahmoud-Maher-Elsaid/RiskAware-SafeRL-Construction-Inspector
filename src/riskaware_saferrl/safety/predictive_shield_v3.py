from __future__ import annotations

import time
from dataclasses import asdict, dataclass

import numpy as np

from riskaware_saferrl.safety.safety_contract_v3 import (
    ControlledInspectionState,
    SafetyContractV3,
    SafetyVectorCost,
)

Position = tuple[int, int]
ACTION_DELTAS = {
    0: (-1, 0),
    1: (1, 0),
    2: (0, -1),
    3: (0, 1),
}


@dataclass(frozen=True)
class EventAwareShieldDecisionV3:
    proposed_action: int
    final_action: int
    shield_decision: str
    rejection_reasons: tuple[str, ...]
    predicted_trajectory: tuple[Position, ...]
    predicted_vector_cost: SafetyVectorCost
    proposed_trajectory: tuple[Position, ...]
    proposed_vector_cost: SafetyVectorCost
    adaptive_horizon: int
    human_clearance_margin: float
    controlled_inspection: bool
    emergency_stop: bool
    recovery_active: bool
    cooldown_remaining: int
    computation_time_ms: float

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


class EventAwarePredictiveShieldV3:
    """Observation-only vector-cost shield with bounded intervention memory."""

    def __init__(
        self,
        contract: SafetyContractV3,
        *,
        minimum_horizon: int = 1,
        maximum_horizon: int = 5,
        intervention_cooldown: int = 2,
        hysteresis_margin: float = 0.1,
        action_continuity_penalty: float = 0.05,
        emergency_action: int = 4,
    ) -> None:
        if not 1 <= minimum_horizon <= maximum_horizon <= 5:
            raise ValueError("Shield horizon must satisfy 1 <= minimum <= maximum <= 5")
        self.contract = contract
        self.minimum_horizon = minimum_horizon
        self.maximum_horizon = maximum_horizon
        self.intervention_cooldown = intervention_cooldown
        self.hysteresis_margin = hysteresis_margin
        self.action_continuity_penalty = action_continuity_penalty
        self.emergency_action = emergency_action
        self.reset()

    def reset(self) -> None:
        self._previous_final_action: int | None = None
        self._cooldown_remaining = 0
        self._recent_positions: list[Position] = []
        self._recent_interventions: list[bool] = []
        self._emergency_active = False

    @staticmethod
    def _robot_position(semantic_map: np.ndarray) -> Position:
        locations = np.argwhere(semantic_map[5] > 0)
        if len(locations) != 1:
            raise ValueError("Shield observation must contain exactly one robot position")
        return int(locations[0, 0]), int(locations[0, 1])

    @staticmethod
    def _inside(position: Position, size: int) -> bool:
        return 0 <= position[0] < size and 0 <= position[1] < size

    def _adaptive_horizon(self, semantic_map: np.ndarray, position: Position) -> int:
        risk_locations = np.argwhere(
            (semantic_map[2] > 0)
            | (semantic_map[3] > 0)
            | (semantic_map[7] > 0)
            | (semantic_map[8] > 0)
        )
        if len(risk_locations) == 0:
            return self.minimum_horizon
        distance = min(
            abs(position[0] - int(row)) + abs(position[1] - int(column))
            for row, column in risk_locations
        )
        return max(
            self.minimum_horizon,
            min(self.maximum_horizon, self.maximum_horizon - distance + 2),
        )

    @staticmethod
    def _clearance(position: Position, workers: np.ndarray) -> float:
        locations = np.argwhere(workers > 0)
        if len(locations) == 0:
            return float("inf")
        return float(
            min(
                abs(position[0] - int(row)) + abs(position[1] - int(column))
                for row, column in locations
            )
        )

    def _predict(
        self,
        semantic_map: np.ndarray,
        action_mask: np.ndarray,
        action: int,
        horizon: int,
        inspection: ControlledInspectionState,
    ) -> tuple[tuple[Position, ...], SafetyVectorCost, float, tuple[str, ...]]:
        size = semantic_map.shape[-1]
        position = self._robot_position(semantic_map)
        trajectory = [position]
        collision = restricted = worker_breach = semantic_risk = uncertainty = 0.0
        reasons: list[str] = []
        if action == self.emergency_action:
            controlled = self.contract.controlled_inspection_valid(inspection)
            semantic_risk = 0.0 if controlled else float(semantic_map[8, position[0], position[1]])
            clearance = self._clearance(position, semantic_map[2])
            minimum_clearance = float(
                self.contract.config["controlled_inspection"]["minimum_human_clearance_cells"]
            )
            worker_breach = float(clearance < minimum_clearance)
            vector = self.contract.vector_cost(
                collision=False,
                restricted_zone=bool(semantic_map[3, position[0], position[1]]),
                human_clearance_breach=bool(worker_breach),
                semantic_risk=semantic_risk,
                uncertainty=0.0,
                inspection=inspection,
            )
            return tuple(trajectory), vector, clearance, tuple(reasons)
        if not bool(action_mask[action]):
            collision = 1.0
            reasons.append("invalid_or_obstructed_action")
        dynamic = semantic_map[7] > 0
        for prediction_step in range(horizon):
            if collision:
                break
            delta = ACTION_DELTAS[action]
            candidate = position[0] + delta[0], position[1] + delta[1]
            if not self._inside(candidate, size) or semantic_map[0, candidate[0], candidate[1]] > 0:
                if prediction_step == 0:
                    collision = 1.0
                    reasons.append("predicted_collision")
                else:
                    uncertainty = max(uncertainty, 0.5)
                    reasons.append("future_collision_risk")
                break
            position = candidate
            trajectory.append(position)
            if semantic_map[3, position[0], position[1]] > 0:
                if prediction_step == 0:
                    restricted = 1.0
                    reasons.append("restricted_zone")
                else:
                    semantic_risk = max(semantic_risk, 1.0)
                    reasons.append("future_restricted_zone_risk")
            clearance = self._clearance(position, semantic_map[2])
            minimum_clearance = float(
                self.contract.config["controlled_inspection"]["minimum_human_clearance_cells"]
            )
            if clearance < minimum_clearance:
                if prediction_step == 0:
                    worker_breach = max(worker_breach, 1.0)
                    reasons.append("human_clearance_breach")
                else:
                    semantic_risk = max(semantic_risk, 0.9)
                    reasons.append("future_human_clearance_risk")
            dynamic_locations = np.argwhere(dynamic)
            if any(
                abs(position[0] - int(row)) + abs(position[1] - int(column)) <= prediction_step + 1
                for row, column in dynamic_locations
            ):
                semantic_risk = max(semantic_risk, 0.85)
                uncertainty = max(uncertainty, 0.25 * (prediction_step + 1))
                reasons.append("dynamic_hazard_prediction")
            semantic_risk = max(
                semantic_risk,
                float(semantic_map[8, position[0], position[1]]),
                float(semantic_map[6, position[0], position[1]]),
            )
        clearance = self._clearance(position, semantic_map[2])
        vector = self.contract.vector_cost(
            collision=bool(collision),
            restricted_zone=bool(restricted),
            human_clearance_breach=bool(worker_breach),
            semantic_risk=semantic_risk,
            uncertainty=uncertainty,
            inspection=inspection,
        )
        return (
            tuple(trajectory),
            vector,
            clearance,
            tuple(dict.fromkeys(reasons)),
        )

    @staticmethod
    def _hard_cost(vector: SafetyVectorCost) -> float:
        return vector.collision + vector.restricted_zone + vector.worker_near_miss

    @staticmethod
    def _soft_cost(vector: SafetyVectorCost) -> float:
        return vector.uncontrolled_semantic_risk + vector.uncertainty

    def decide(
        self,
        observation: dict[str, np.ndarray],
        proposed_action: int,
        inspection: ControlledInspectionState | None = None,
    ) -> EventAwareShieldDecisionV3:
        started = time.perf_counter()
        semantic_map = np.asarray(observation["map"])
        action_mask = np.asarray(observation["action_mask"], dtype=np.bool_)
        position = self._robot_position(semantic_map)
        self._recent_positions.append(position)
        self._recent_positions = self._recent_positions[-10:]
        deadlocked = (
            len(self._recent_positions) == 10
            and len(set(self._recent_positions)) <= 2
            and sum(self._recent_interventions[-10:]) >= 6
        )
        inspection = inspection or ControlledInspectionState(
            human_clearance_cells=float(
                self.contract.config["controlled_inspection"]["minimum_human_clearance_cells"]
            )
        )
        horizon = self._adaptive_horizon(semantic_map, position)
        proposed = self._predict(
            semantic_map,
            action_mask,
            proposed_action,
            horizon,
            inspection,
        )
        proposed_hard = self._hard_cost(proposed[1])
        requires_intervention = proposed_hard > 0.0
        if self._cooldown_remaining > 0:
            self._cooldown_remaining -= 1
        candidates = []
        for action in range(len(action_mask)):
            if not bool(action_mask[action]):
                continue
            prediction = self._predict(
                semantic_map,
                action_mask,
                action,
                horizon,
                inspection,
            )
            continuity = (
                self.action_continuity_penalty
                if self._previous_final_action is not None and action != self._previous_final_action
                else 0.0
            )
            progress_penalty = 0.5 if action == self.emergency_action else 0.0
            if deadlocked and action == self.emergency_action:
                progress_penalty += 2.0
            candidates.append(
                (
                    self._hard_cost(prediction[1]),
                    self._soft_cost(prediction[1]),
                    progress_penalty + continuity + 0.01 * abs(action - proposed_action),
                    action,
                    prediction,
                )
            )
        best = min(candidates)
        controlled = self.contract.controlled_inspection_valid(inspection)
        final_action = proposed_action
        decision = "accept"
        emergency = False
        recovery = False
        selected = proposed
        if requires_intervention:
            final_action = best[3]
            selected = best[4]
            if best[0] == 0.0:
                decision = "retreat" if deadlocked else "replace"
                recovery = deadlocked
            elif final_action == self.emergency_action:
                decision = "stop"
                emergency = True
            else:
                decision = "modify"
            self._cooldown_remaining = self.intervention_cooldown
        elif deadlocked and best[3] != self.emergency_action:
            final_action = best[3]
            selected = best[4]
            decision = "retreat"
            recovery = True
        self._emergency_active = emergency
        self._previous_final_action = final_action
        self._recent_interventions.append(decision != "accept")
        self._recent_interventions = self._recent_interventions[-10:]
        elapsed = (time.perf_counter() - started) * 1000.0
        return EventAwareShieldDecisionV3(
            proposed_action=proposed_action,
            final_action=final_action,
            shield_decision=decision,
            rejection_reasons=selected[3] if decision == "accept" else proposed[3],
            predicted_trajectory=selected[0],
            predicted_vector_cost=selected[1],
            proposed_trajectory=proposed[0],
            proposed_vector_cost=proposed[1],
            adaptive_horizon=horizon,
            human_clearance_margin=selected[2],
            controlled_inspection=controlled,
            emergency_stop=emergency,
            recovery_active=recovery,
            cooldown_remaining=self._cooldown_remaining,
            computation_time_ms=elapsed,
        )
