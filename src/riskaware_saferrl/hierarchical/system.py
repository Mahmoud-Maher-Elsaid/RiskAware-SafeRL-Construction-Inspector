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
        shield_result = self.shield.decide(observation, controller_result.primitive, inspection)
        return HierarchicalDecision(
            option=option,
            planner=planner_result,
            controller=controller_result,
            shield=shield_result,
            executed_primitive=shield_result.final_action,
        )
