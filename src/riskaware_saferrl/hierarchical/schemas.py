from __future__ import annotations

from enum import IntEnum

import numpy as np


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


def causal_option_mask(
    semantic_map: np.ndarray,
    primitive_action_mask: np.ndarray | None = None,
) -> np.ndarray:
    """Return mission options feasible under the current causal observation."""
    if semantic_map.shape[0] < 11:
        raise ValueError("The hierarchical option mask requires 11 semantic channels")
    robot_locations = np.argwhere(semantic_map[5] > 0)
    if len(robot_locations) != 1:
        raise ValueError("The semantic map must contain exactly one robot cell")
    robot = robot_locations[0]
    workers = np.argwhere((semantic_map[2] > 0) | (semantic_map[7] > 0))
    worker_distance = min(
        (int(np.abs(worker - robot).sum()) for worker in workers),
        default=99,
    )
    uninspected = semantic_map[10] <= 0
    targets = np.argwhere((semantic_map[1] > 0) & uninspected)
    nearest: tuple[int, int] | None = None
    if len(targets):
        nearest = min(
            ((int(row), int(column)) for row, column in targets),
            key=lambda value: (
                abs(value[0] - int(robot[0])) + abs(value[1] - int(robot[1])),
                value,
            ),
        )
    mask = np.zeros(OPTION_COUNT, dtype=np.bool_)
    mask[MissionOption.REPLAN_ROUTE] = True
    mask[MissionOption.HOLD_FOR_UNCERTAINTY] = True
    mask[MissionOption.EMERGENCY_SAFE_STOP] = True
    if worker_distance <= 1:
        primitive_mask = (
            np.ones(5, dtype=np.bool_)
            if primitive_action_mask is None
            else np.asarray(primitive_action_mask, dtype=np.bool_)
        )
        deltas = ((-1, 0), (1, 0), (0, -1), (0, 1), (0, 0))
        escape_available = any(
            bool(primitive_mask[action])
            and min(
                (
                    abs(int(robot[0]) + delta[0] - int(worker[0]))
                    + abs(int(robot[1]) + delta[1] - int(worker[1]))
                    for worker in workers
                ),
                default=99,
            )
            > worker_distance
            for action, delta in enumerate(deltas)
        )
        if escape_available:
            mask[MissionOption.AVOID_DYNAMIC_WORKER] = True
            mask[MissionOption.RETREAT_TO_SAFE_CELL] = True
            mask[MissionOption.HOLD_FOR_UNCERTAINTY] = False
        return mask
    mask[MissionOption.EXPLORE_FRONTIER] = True
    primitive_mask = (
        np.ones(5, dtype=np.bool_)
        if primitive_action_mask is None
        else np.asarray(primitive_action_mask, dtype=np.bool_)
    )
    if nearest is not None:
        mask[MissionOption.CONTINUE_CURRENT_TARGET] = True
        if bool(primitive_mask[4]):
            mask[MissionOption.INSPECT_KNOWN_RISK] = True
    return mask
