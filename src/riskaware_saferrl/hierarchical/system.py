from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from riskaware_saferrl.hierarchical.controller import (
    LocalControllerDecision,
    PredictiveLocalController,
)
from riskaware_saferrl.hierarchical.planner import (
    CausalRiskAwarePlanner,
    PlannerRequest,
    PlannerResult,
)
from riskaware_saferrl.hierarchical.schemas import MissionOption
from riskaware_saferrl.safety import (
    ControlledInspectionState,
    EventAwarePredictiveShieldV3,
)
from riskaware_saferrl.safety.predictive_shield_v3 import EventAwareShieldDecisionV3

_CANONICAL_TO_SHIELD = {0: 4, 1: 0, 2: 2, 3: 3, 4: 4}
_SHIELD_TO_CANONICAL = {0: 1, 2: 2, 3: 3, 4: 0}


@dataclass(frozen=True)
class HierarchicalDecision:
    option: MissionOption
    planner: PlannerResult
    controller: LocalControllerDecision
    shield: EventAwareShieldDecisionV3
    executed_primitive: int


class RiskShieldHierarchicalSystem:
    """Causal planner/controller/shield execution below a learned option policy."""

    def __init__(
        self,
        planner: CausalRiskAwarePlanner,
        controller: PredictiveLocalController,
        shield: EventAwarePredictiveShieldV3,
    ) -> None:
        self.planner = planner
        self.controller = controller
        self.shield = shield

    def reset(self) -> None:
        self.planner.reset()
        self.controller.reset()
        self.shield.reset()

    def execute_option(
        self,
        observation: dict[str, np.ndarray],
        *,
        option: MissionOption,
        target: tuple[int, int] | None,
        risk_budget: float,
        inspection_intent: bool,
        force_replan: bool = False,
    ) -> HierarchicalDecision:
        planner_result = self.planner.plan(
            observation,
            PlannerRequest(
                option=option,
                target=target,
                risk_budget=float(np.clip(risk_budget, 0.0, 1.0)),
                inspection_intent=inspection_intent,
                force_replan=force_replan,
            ),
        )
        controller_result = self.controller.decide(
            observation,
            planner_result.path,
            inspection_intent=inspection_intent,
        )
        inspection = ControlledInspectionState(
            target_recognized=target is not None,
            inspection_intent=inspection_intent,
            safe_approach=planner_result.success,
            speed=0.0 if controller_result.primitive == 4 else 1.0,
            human_clearance_cells=controller_result.human_clearance_margin,
            dwell_steps=1,
            retreat_route_available=planner_result.success,
            shield_active=True,
        )
        # The shield's historical action contract is sensor/grid based
        # (0=forward, 2=left, 3=right, 4=emergency), while the hierarchical
        # controller uses the canonical primitive enum (0=stop, 1=forward,
        # 2=left, 3=right, 4=inspect). Translate at this single boundary so a
        # shield recovery action 0 cannot be misinterpreted as canonical STOP.
        shield_action = _CANONICAL_TO_SHIELD[controller_result.primitive]
        shield_result_grid = self.shield.decide(observation, shield_action, inspection)
        canonical_final = _SHIELD_TO_CANONICAL.get(shield_result_grid.final_action, 0)
        canonical_proposed = _SHIELD_TO_CANONICAL.get(shield_result_grid.proposed_action, 0)
        shield_result = EventAwareShieldDecisionV3(
            proposed_action=canonical_proposed,
            final_action=canonical_final,
            shield_decision=shield_result_grid.shield_decision,
            rejection_reasons=shield_result_grid.rejection_reasons,
            predicted_trajectory=shield_result_grid.predicted_trajectory,
            predicted_vector_cost=shield_result_grid.predicted_vector_cost,
            proposed_trajectory=shield_result_grid.proposed_trajectory,
            proposed_vector_cost=shield_result_grid.proposed_vector_cost,
            adaptive_horizon=shield_result_grid.adaptive_horizon,
            human_clearance_margin=shield_result_grid.human_clearance_margin,
            controlled_inspection=shield_result_grid.controlled_inspection,
            emergency_stop=shield_result_grid.emergency_stop,
            recovery_active=shield_result_grid.recovery_active,
            cooldown_remaining=shield_result_grid.cooldown_remaining,
            computation_time_ms=shield_result_grid.computation_time_ms,
        )
        return HierarchicalDecision(
            option=option,
            planner=planner_result,
            controller=controller_result,
            shield=shield_result,
            executed_primitive=shield_result.final_action,
        )
