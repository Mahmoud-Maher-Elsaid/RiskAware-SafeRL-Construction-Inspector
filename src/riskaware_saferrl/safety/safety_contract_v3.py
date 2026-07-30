from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

import yaml

Position = tuple[int, int]
EventSeverity = Literal["hard", "soft", "controlled"]
VECTOR_COST_NAMES = (
    "collision",
    "restricted_zone",
    "worker_near_miss",
    "uncontrolled_semantic_risk",
    "uncertainty",
)


@dataclass(frozen=True)
class ControlledInspectionState:
    target_recognized: bool = False
    inspection_intent: bool = False
    safe_approach: bool = False
    speed: float = 0.0
    human_clearance_cells: float = float("inf")
    dwell_steps: int = 0
    retreat_route_available: bool = False
    shield_active: bool = False


@dataclass(frozen=True)
class SafetyVectorCost:
    collision: float = 0.0
    restricted_zone: float = 0.0
    worker_near_miss: float = 0.0
    uncontrolled_semantic_risk: float = 0.0
    uncertainty: float = 0.0

    def as_tuple(self) -> tuple[float, ...]:
        return tuple(float(getattr(self, name)) for name in VECTOR_COST_NAMES)

    def as_dict(self) -> dict[str, float]:
        return {name: float(getattr(self, name)) for name in VECTOR_COST_NAMES}


@dataclass(frozen=True)
class SafetyEvent:
    event_id: str
    event_type: str
    severity: EventSeverity
    start_step: int
    end_step: int
    duration: int
    peak_severity: float
    integrated_severity: float
    robot_position: Position
    target_identifier: str | None
    proposed_action: int
    executed_action: int
    shield_decision: str
    inspection_intent: bool
    controlled_exposure: bool
    resolved: bool
    resolution_action: int | None


@dataclass(frozen=True)
class SafetyTransitionRecord:
    observation: dict[str, Any]
    policy_action: int
    action_mask: tuple[bool, ...]
    predicted_trajectory: tuple[Position, ...]
    predicted_vector_cost: SafetyVectorCost
    shield_result: dict[str, Any]
    executed_action: int
    next_state: dict[str, Any]
    actual_vector_cost: SafetyVectorCost
    event_updates: tuple[SafetyEvent, ...]
    terminated: bool
    truncated: bool
    legacy_raw_safety_cost: float


@dataclass
class _OpenEvent:
    event_id: str
    event_type: str
    severity: EventSeverity
    start_step: int
    last_step: int
    duration: int
    peak_severity: float
    integrated_severity: float
    robot_position: Position
    target_identifier: str | None
    proposed_action: int
    executed_action: int
    shield_decision: str
    inspection_intent: bool
    controlled_exposure: bool


@dataclass
class SafetyContractMetrics:
    legacy_raw_safety_cost: float = 0.0
    legacy_constraint_count: int = 0
    steps: int = 0
    mission_progress: float = 0.0
    success: bool = False
    events: list[SafetyEvent] = field(default_factory=list)


def load_safety_contract(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if payload.get("version") != "3.0":
        raise ValueError("Safety Contract v3 requires version 3.0")
    required = {
        "hard_constraints",
        "soft_constraints",
        "controlled_inspection",
        "event_tracking",
        "legacy",
    }
    missing = required - payload.keys()
    if missing:
        raise ValueError(f"Safety Contract v3 is missing: {sorted(missing)}")
    return payload


class SafetyContractV3:
    """Event-aware safety accounting that preserves all legacy metrics."""

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        self.reset()

    @classmethod
    def from_yaml(cls, path: Path) -> SafetyContractV3:
        return cls(load_safety_contract(path))

    def reset(self) -> None:
        self.metrics = SafetyContractMetrics()
        self._open: dict[str, _OpenEvent] = {}
        self._event_indices: dict[str, int] = {}

    def controlled_inspection_valid(self, state: ControlledInspectionState) -> bool:
        config = self.config["controlled_inspection"]
        return bool(
            state.target_recognized
            and state.inspection_intent
            and state.safe_approach
            and state.speed <= float(config["maximum_motion_speed"])
            and state.human_clearance_cells >= float(config["minimum_human_clearance_cells"])
            and state.dwell_steps <= int(config["maximum_dwell_steps"])
            and state.retreat_route_available
            and state.shield_active
        )

    def vector_cost(
        self,
        *,
        collision: bool,
        restricted_zone: bool,
        human_clearance_breach: bool,
        semantic_risk: float,
        uncertainty: float,
        inspection: ControlledInspectionState,
    ) -> SafetyVectorCost:
        controlled = self.controlled_inspection_valid(inspection)
        return SafetyVectorCost(
            collision=float(collision),
            restricted_zone=float(restricted_zone),
            worker_near_miss=float(human_clearance_breach),
            uncontrolled_semantic_risk=0.0 if controlled else max(0.0, semantic_risk),
            uncertainty=max(0.0, uncertainty),
        )

    def update_events(
        self,
        *,
        episode_id: str,
        step: int,
        position: Position,
        active: dict[str, tuple[EventSeverity, float, bool]],
        proposed_action: int,
        executed_action: int,
        shield_decision: str,
        inspection_intent: bool,
        target_identifier: str | None = None,
    ) -> tuple[SafetyEvent, ...]:
        closed: list[SafetyEvent] = []
        for event_type in tuple(self._open):
            if event_type not in active:
                closed.append(
                    self._close(
                        event_type,
                        resolved=True,
                        resolution_action=executed_action,
                    )
                )
        for event_type, (severity, value, controlled) in active.items():
            if event_type not in self._open:
                index = self._event_indices.get(event_type, 0) + 1
                self._event_indices[event_type] = index
                self._open[event_type] = _OpenEvent(
                    event_id=f"{episode_id}:{event_type}:{index:05d}",
                    event_type=event_type,
                    severity=severity,
                    start_step=step,
                    last_step=step,
                    duration=0,
                    peak_severity=0.0,
                    integrated_severity=0.0,
                    robot_position=position,
                    target_identifier=target_identifier,
                    proposed_action=proposed_action,
                    executed_action=executed_action,
                    shield_decision=shield_decision,
                    inspection_intent=inspection_intent,
                    controlled_exposure=controlled,
                )
            event = self._open[event_type]
            event.last_step = step
            event.duration += 1
            event.peak_severity = max(event.peak_severity, value)
            event.integrated_severity += value
            event.controlled_exposure &= controlled
        self.metrics.events.extend(closed)
        return tuple(closed)

    def _close(
        self, event_type: str, *, resolved: bool, resolution_action: int | None
    ) -> SafetyEvent:
        event = self._open.pop(event_type)
        return SafetyEvent(
            event_id=event.event_id,
            event_type=event.event_type,
            severity=event.severity,
            start_step=event.start_step,
            end_step=event.last_step,
            duration=event.duration,
            peak_severity=event.peak_severity,
            integrated_severity=event.integrated_severity,
            robot_position=event.robot_position,
            target_identifier=event.target_identifier,
            proposed_action=event.proposed_action,
            executed_action=event.executed_action,
            shield_decision=event.shield_decision,
            inspection_intent=event.inspection_intent,
            controlled_exposure=event.controlled_exposure,
            resolved=resolved,
            resolution_action=resolution_action,
        )

    def finish_episode(
        self, *, success: bool, mission_progress: float, resolution_action: int | None
    ) -> tuple[SafetyEvent, ...]:
        closed = tuple(
            self._close(
                event_type,
                resolved=success,
                resolution_action=resolution_action,
            )
            for event_type in tuple(self._open)
        )
        self.metrics.events.extend(closed)
        self.metrics.success = success
        self.metrics.mission_progress = mission_progress
        return closed

    def add_legacy(self, *, raw_cost: float, constraint_count: int) -> None:
        self.metrics.legacy_raw_safety_cost += raw_cost
        self.metrics.legacy_constraint_count += constraint_count
        self.metrics.steps += 1

    def summary(self) -> dict[str, Any]:
        hard_events = [event for event in self.metrics.events if event.severity == "hard"]
        soft_events = [event for event in self.metrics.events if event.severity == "soft"]
        controlled_events = [
            event for event in self.metrics.events if event.severity == "controlled"
        ]
        hard_cost = sum(event.integrated_severity for event in hard_events)
        soft_cost = sum(event.integrated_severity for event in soft_events)
        event_cost = (
            hard_cost + soft_cost + sum(event.integrated_severity for event in controlled_events)
        )
        steps = max(1, self.metrics.steps)
        progress = self.metrics.mission_progress
        return {
            "legacy_raw_safety_cost": self.metrics.legacy_raw_safety_cost,
            "legacy_constraint_count": self.metrics.legacy_constraint_count,
            "hard_safety_cost_v3": hard_cost,
            "soft_safety_cost_v3": soft_cost,
            "event_safety_cost_v3": event_cost,
            "success_conditioned_safety_cost_v3": (event_cost if self.metrics.success else None),
            "mission_progress_normalized_cost_v3": (
                event_cost / progress if progress > 0.0 else None
            ),
            "cost_per_100_steps_v3": 100.0 * event_cost / steps,
            "hard_violations_per_episode": len(hard_events),
            "soft_events_per_episode": len(soft_events),
            "controlled_exposure_events": len(controlled_events),
            "events": [asdict(event) for event in self.metrics.events],
            "legacy_diagnostic_label": "LEGACY_INACTIVITY_BIASED_DIAGNOSTIC",
        }
