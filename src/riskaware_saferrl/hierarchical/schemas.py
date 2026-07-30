from __future__ import annotations

from enum import IntEnum


class MissionOption(IntEnum):
    """Learned mission decisions; primitives remain the planner's responsibility."""

    EXPLORE_FRONTIER = 0
    INSPECT_KNOWN_RISK = 1
    INSPECT_PPE_VIOLATION = 2
    CONTINUE_CURRENT_TARGET = 3
    AVOID_DYNAMIC_WORKER = 4
    RETREAT_TO_SAFE_CELL = 5
    REPLAN_ROUTE = 6
    HOLD_FOR_UNCERTAINTY = 7
    EMERGENCY_SAFE_STOP = 8


OPTION_COUNT = len(MissionOption)
PRIMITIVE_COUNT = 5
VECTOR_COST_NAMES = (
    "collision",
    "restricted_zone",
    "human_clearance",
    "uncontrolled_semantic_risk",
    "uncertainty",
    "shield_dependence",
    "deadlock",
)
