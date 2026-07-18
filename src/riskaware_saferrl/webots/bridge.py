from __future__ import annotations

import math
from dataclasses import dataclass
from enum import IntEnum

import numpy as np

from riskaware_saferrl.inspection import (
    inspectable_hazards_from,
)
from riskaware_saferrl.webots.motion_primitives import (
    MotionPrimitive,
)

GridPosition = tuple[int, int]


class GridAction(IntEnum):
    MOVE_UP = 0
    MOVE_DOWN = 1
    MOVE_LEFT = 2
    MOVE_RIGHT = 3
    INSPECT = 4


class CardinalHeading(IntEnum):
    EAST = 0
    NORTH = 1
    WEST = 2
    SOUTH = 3

    @classmethod
    def from_yaw(
        cls,
        yaw_radians: float,
    ) -> CardinalHeading:
        if not math.isfinite(yaw_radians):
            raise ValueError("yaw_radians must be finite.")

        quarter_turn = int(round(yaw_radians / (math.pi / 2.0))) % 4

        return cls(quarter_turn)


ACTION_TO_DELTA: dict[
    GridAction,
    GridPosition,
] = {
    GridAction.MOVE_UP: (-1, 0),
    GridAction.MOVE_DOWN: (1, 0),
    GridAction.MOVE_LEFT: (0, -1),
    GridAction.MOVE_RIGHT: (0, 1),
}


ACTION_TO_HEADING: dict[
    GridAction,
    CardinalHeading,
] = {
    GridAction.MOVE_UP: CardinalHeading.NORTH,
    GridAction.MOVE_DOWN: CardinalHeading.SOUTH,
    GridAction.MOVE_LEFT: CardinalHeading.WEST,
    GridAction.MOVE_RIGHT: CardinalHeading.EAST,
}


@dataclass(frozen=True)
class GridFrame:
    size: int = 12
    x_min: float = -5.0
    x_max: float = 5.0
    z_min: float = -4.0
    z_max: float = 4.0

    def __post_init__(self) -> None:
        if self.size < 2:
            raise ValueError("size must be at least 2.")

        if self.x_min >= self.x_max:
            raise ValueError("x_min must be smaller than x_max.")

        if self.z_min >= self.z_max:
            raise ValueError("z_min must be smaller than z_max.")

    @property
    def cell_width(self) -> float:
        return (self.x_max - self.x_min) / self.size

    @property
    def cell_depth(self) -> float:
        return (self.z_max - self.z_min) / self.size

    def contains_grid_position(
        self,
        position: GridPosition,
    ) -> bool:
        row, column = position

        return 0 <= row < self.size and 0 <= column < self.size

    def world_to_grid(
        self,
        x: float,
        z: float,
    ) -> GridPosition:
        if not math.isfinite(x) or not math.isfinite(z):
            raise ValueError("World coordinates must be finite.")

        if not self.x_min <= x <= self.x_max:
            raise ValueError(f"x coordinate is outside the grid: {x}")

        if not self.z_min <= z <= self.z_max:
            raise ValueError(f"z coordinate is outside the grid: {z}")

        column = int(math.floor((x - self.x_min) / self.cell_width))

        row = int(math.floor((self.z_max - z) / self.cell_depth))

        row = min(
            self.size - 1,
            max(0, row),
        )

        column = min(
            self.size - 1,
            max(0, column),
        )

        return row, column

    def grid_to_world(
        self,
        position: GridPosition,
    ) -> tuple[float, float]:
        if not self.contains_grid_position(position):
            raise ValueError(f"Grid position is outside the frame: {position}")

        row, column = position

        x = self.x_min + (column + 0.5) * self.cell_width

        z = self.z_max - (row + 0.5) * self.cell_depth

        return float(x), float(z)


@dataclass(frozen=True)
class WebotsSensorSnapshot:
    x: float
    z: float
    yaw_radians: float
    proximity: tuple[float, ...]
    step_count: int = 0
    max_steps: int = 250
    inspected_ratio: float = 0.0

    def __post_init__(self) -> None:
        numeric_values = (
            self.x,
            self.z,
            self.yaw_radians,
            *self.proximity,
        )

        if not all(math.isfinite(value) for value in numeric_values):
            raise ValueError("Sensor snapshot values must be finite.")

        if self.step_count < 0:
            raise ValueError("step_count must be non-negative.")

        if self.max_steps <= 0:
            raise ValueError("max_steps must be positive.")

        if not (0.0 <= self.inspected_ratio <= 1.0):
            raise ValueError("inspected_ratio must be between 0 and 1.")

    @property
    def heading(self) -> CardinalHeading:
        return CardinalHeading.from_yaw(self.yaw_radians)

    def grid_position(
        self,
        frame: GridFrame,
    ) -> GridPosition:
        return frame.world_to_grid(
            self.x,
            self.z,
        )

    def normalized_proximity(
        self,
        *,
        maximum_reading: float = 4095.0,
    ) -> np.ndarray:
        if maximum_reading <= 0.0:
            raise ValueError("maximum_reading must be positive.")

        readings = np.asarray(
            self.proximity,
            dtype=np.float32,
        )

        return np.clip(
            readings / maximum_reading,
            0.0,
            1.0,
        ).astype(np.float32)


@dataclass(frozen=True)
class SemanticScene:
    size: int
    obstacles: frozenset[GridPosition] = frozenset()
    hazards: frozenset[GridPosition] = frozenset()
    workers: frozenset[GridPosition] = frozenset()
    restricted: frozenset[GridPosition] = frozenset()

    def __post_init__(self) -> None:
        if self.size < 2:
            raise ValueError("size must be at least 2.")

        groups = (
            ("obstacles", self.obstacles),
            ("hazards", self.hazards),
            ("workers", self.workers),
            ("restricted", self.restricted),
        )

        for group_name, positions in groups:
            for position in positions:
                row, column = position

                if not (0 <= row < self.size and 0 <= column < self.size):
                    raise ValueError(f"{group_name} contains an invalid position: {position}")


@dataclass(frozen=True)
class BridgeState:
    agent_position: GridPosition
    visited: frozenset[GridPosition]
    inspected: frozenset[GridPosition]
    steps: int
    max_steps: int

    def __post_init__(self) -> None:
        if self.steps < 0:
            raise ValueError("steps must be non-negative.")

        if self.max_steps <= 0:
            raise ValueError("max_steps must be positive.")


@dataclass(frozen=True)
class ActionBridge:
    def plan(
        self,
        action: GridAction | int,
        current_heading: CardinalHeading,
    ) -> tuple[MotionPrimitive, ...]:
        try:
            resolved_action = GridAction(int(action))
        except (
            TypeError,
            ValueError,
        ) as error:
            raise ValueError(f"Unsupported grid action: {action!r}") from error

        if resolved_action is GridAction.INSPECT:
            return (MotionPrimitive.INSPECT,)

        target_heading = ACTION_TO_HEADING[resolved_action]

        rotation_delta = (int(target_heading) - int(current_heading)) % 4

        if rotation_delta == 0:
            return (MotionPrimitive.MOVE_FORWARD,)

        if rotation_delta == 1:
            return (
                MotionPrimitive.TURN_LEFT,
                MotionPrimitive.MOVE_FORWARD,
            )

        if rotation_delta == 3:
            return (
                MotionPrimitive.TURN_RIGHT,
                MotionPrimitive.MOVE_FORWARD,
            )

        return (
            MotionPrimitive.TURN_LEFT,
            MotionPrimitive.TURN_LEFT,
            MotionPrimitive.MOVE_FORWARD,
        )


@dataclass(frozen=True)
class ObservationBridge:
    vision_radius: int = 3
    inspection_radius: int = 2

    def __post_init__(self) -> None:
        if self.vision_radius < 0:
            raise ValueError("vision_radius must be non-negative.")

        if self.inspection_radius < 0:
            raise ValueError("inspection_radius must be non-negative.")

    @staticmethod
    def _contains(
        scene: SemanticScene,
        position: GridPosition,
    ) -> bool:
        row, column = position

        return 0 <= row < scene.size and 0 <= column < scene.size

    def _validate_state(
        self,
        scene: SemanticScene,
        state: BridgeState,
    ) -> None:
        positions = (
            state.agent_position,
            *state.visited,
            *state.inspected,
        )

        for position in positions:
            if not self._contains(
                scene,
                position,
            ):
                raise ValueError(f"Bridge state contains an invalid position: {position}")

        if not state.inspected.issubset(scene.hazards):
            raise ValueError("Inspected positions must be a subset of scene hazards.")

    def _visible(
        self,
        agent_position: GridPosition,
        position: GridPosition,
    ) -> bool:
        distance = abs(position[0] - agent_position[0]) + abs(position[1] - agent_position[1])

        return distance <= self.vision_radius

    @staticmethod
    def _paint_local_risk(
        risk_map: np.ndarray,
        center: GridPosition,
        value: float,
    ) -> None:
        offsets = (
            (0, 0),
            (-1, 0),
            (1, 0),
            (0, -1),
            (0, 1),
        )

        for row_delta, column_delta in offsets:
            row = center[0] + row_delta
            column = center[1] + column_delta

            if 0 <= row < risk_map.shape[0] and 0 <= column < risk_map.shape[1]:
                risk_map[row, column] = max(
                    risk_map[row, column],
                    value,
                )

    def inspectable_hazards(
        self,
        scene: SemanticScene,
        state: BridgeState,
    ) -> frozenset[GridPosition]:
        self._validate_state(
            scene,
            state,
        )

        return inspectable_hazards_from(
            state.agent_position,
            scene.hazards - state.inspected,
            blockers=scene.obstacles,
            inspection_radius=(self.inspection_radius),
        )

    def build_observation(
        self,
        scene: SemanticScene,
        state: BridgeState,
    ) -> dict[str, np.ndarray]:
        self._validate_state(
            scene,
            state,
        )

        semantic_map = np.zeros(
            (
                7,
                scene.size,
                scene.size,
            ),
            dtype=np.float32,
        )

        for position in scene.obstacles:
            if self._visible(
                state.agent_position,
                position,
            ):
                semantic_map[
                    0,
                    position[0],
                    position[1],
                ] = 1.0

        for position in scene.hazards - state.inspected:
            if self._visible(
                state.agent_position,
                position,
            ):
                semantic_map[
                    1,
                    position[0],
                    position[1],
                ] = 1.0

                semantic_map[
                    6,
                    position[0],
                    position[1],
                ] = 0.7

        for position in scene.workers:
            if self._visible(
                state.agent_position,
                position,
            ):
                semantic_map[
                    2,
                    position[0],
                    position[1],
                ] = 1.0

                self._paint_local_risk(
                    semantic_map[6],
                    position,
                    0.85,
                )

        for position in scene.restricted:
            if self._visible(
                state.agent_position,
                position,
            ):
                semantic_map[
                    3,
                    position[0],
                    position[1],
                ] = 1.0

                semantic_map[
                    6,
                    position[0],
                    position[1],
                ] = 1.0

        for position in state.visited:
            semantic_map[
                4,
                position[0],
                position[1],
            ] = 1.0

        agent_row, agent_column = state.agent_position

        semantic_map[
            5,
            agent_row,
            agent_column,
        ] = 1.0

        normalized_steps = min(
            1.0,
            state.steps
            / max(
                1,
                state.max_steps,
            ),
        )

        inspected_ratio = min(
            1.0,
            len(state.inspected)
            / max(
                1,
                len(scene.hazards),
            ),
        )

        state_vector = np.asarray(
            [
                agent_row
                / max(
                    1,
                    scene.size - 1,
                ),
                agent_column
                / max(
                    1,
                    scene.size - 1,
                ),
                normalized_steps,
                inspected_ratio,
            ],
            dtype=np.float32,
        )

        return {
            "map": semantic_map.reshape(-1),
            "state": state_vector,
        }

    def action_mask(
        self,
        scene: SemanticScene,
        state: BridgeState,
    ) -> np.ndarray:
        self._validate_state(
            scene,
            state,
        )

        mask = np.zeros(
            5,
            dtype=np.bool_,
        )

        agent_row, agent_column = state.agent_position

        for action, delta in ACTION_TO_DELTA.items():
            candidate = (
                agent_row + delta[0],
                agent_column + delta[1],
            )

            mask[int(action)] = (
                self._contains(
                    scene,
                    candidate,
                )
                and candidate not in scene.obstacles
            )

        mask[int(GridAction.INSPECT)] = bool(
            self.inspectable_hazards(
                scene,
                state,
            )
        )

        if not bool(np.any(mask)):
            mask[int(GridAction.INSPECT)] = True

        return mask
