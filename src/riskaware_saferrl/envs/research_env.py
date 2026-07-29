from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

Position = tuple[int, int]


@dataclass(frozen=True)
class GridEnvironmentConfig:
    """Serializable configuration for the final research grid environment."""

    size: int = 12
    obstacle_density: float = 0.12
    hazard_density: float = 0.04
    worker_density: float = 0.025
    restricted_density: float = 0.04
    dynamic_hazard_density: float = 0.015
    ppe_risk_density: float = 0.02
    perception_false_negative_rate: float = 0.0
    vision_radius: int = 4
    inspection_radius: int = 2
    max_steps: int = 250
    dynamic_obstacles: bool = True

    def validate(self) -> None:
        if self.size < 6:
            raise ValueError("size must be at least 6")
        for name in (
            "obstacle_density",
            "hazard_density",
            "worker_density",
            "restricted_density",
            "dynamic_hazard_density",
            "ppe_risk_density",
        ):
            value = float(getattr(self, name))
            if not 0.0 <= value <= 0.5:
                raise ValueError(f"{name} must be in [0, 0.5]")
        if not 0.0 <= self.perception_false_negative_rate <= 1.0:
            raise ValueError("perception_false_negative_rate must be in [0, 1]")
        if self.vision_radius < 1:
            raise ValueError("vision_radius must be positive")
        if self.inspection_radius < 0:
            raise ValueError("inspection_radius must be non-negative")
        if self.max_steps < 1:
            raise ValueError("max_steps must be positive")


@dataclass(frozen=True)
class EpisodeTelemetry:
    episode_seed: int | None
    steps: int
    reward: float
    safety_cost: float
    hazards_inspected: int
    total_hazards: int
    collisions: int
    near_misses: int
    restricted_violations: int
    path_length: float
    energy_usage: float


class ResearchConstructionEnv(gym.Env[dict[str, np.ndarray], int]):
    """Deterministic, partially observable construction inspection benchmark.

    Actions are north, south, west, east, and inspect. Orientation records the
    most recent movement direction and is retained by the inspect action.
    """

    metadata = {"render_modes": ["ansi", "rgb_array"], "render_fps": 4}
    ACTION_NAMES = ("north", "south", "west", "east", "inspect")
    ACTION_TO_DELTA: dict[int, Position] = {
        0: (-1, 0),
        1: (1, 0),
        2: (0, -1),
        3: (0, 1),
    }
    CHANNEL_NAMES = (
        "obstacles",
        "hazards",
        "workers",
        "restricted",
        "visited",
        "agent",
        "risk",
        "dynamic_hazards",
        "ppe_risk",
        "visible",
    )

    def __init__(
        self,
        config: GridEnvironmentConfig | None = None,
        *,
        render_mode: str | None = None,
    ) -> None:
        super().__init__()
        self.config = config or GridEnvironmentConfig()
        self.config.validate()
        self.render_mode = render_mode
        self.size = self.config.size
        self.action_space = spaces.Discrete(len(self.ACTION_NAMES))
        self.observation_space = spaces.Dict(
            {
                "map": spaces.Box(
                    low=0.0,
                    high=1.0,
                    shape=(len(self.CHANNEL_NAMES), self.size, self.size),
                    dtype=np.float32,
                ),
                "state": spaces.Box(low=0.0, high=1.0, shape=(9,), dtype=np.float32),
            }
        )
        self.agent: Position = (0, 0)
        self.orientation = 3
        self.obstacles: set[Position] = set()
        self.hazards: set[Position] = set()
        self.workers: set[Position] = set()
        self.restricted: set[Position] = set()
        self.dynamic_hazards: set[Position] = set()
        self.ppe_risk: set[Position] = set()
        self.inspected: set[Position] = set()
        self.visited: set[Position] = set()
        self.steps = 0
        self.episode_seed: int | None = None
        self.cumulative_reward = 0.0
        self.cumulative_cost = 0.0
        self.collisions = 0
        self.near_misses = 0
        self.restricted_violations = 0
        self.path_length = 0.0
        self.energy_usage = 0.0

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        super().reset(seed=seed)
        self.episode_seed = seed
        self._generate_layout()
        self.inspected = set()
        self.visited = {self.agent}
        self.steps = 0
        self.cumulative_reward = 0.0
        self.cumulative_cost = 0.0
        self.collisions = 0
        self.near_misses = 0
        self.restricted_violations = 0
        self.path_length = 0.0
        self.energy_usage = 0.0
        observation = self._observation()
        return observation, self._info(0.0, False)

    def _entity_count(self, density: float, *, minimum: int = 0) -> int:
        return max(minimum, int(round(self.size * self.size * density)))

    def _generate_layout(self) -> None:
        counts = (
            self._entity_count(self.config.obstacle_density),
            self._entity_count(self.config.hazard_density, minimum=1),
            self._entity_count(self.config.worker_density),
            self._entity_count(self.config.restricted_density),
            self._entity_count(self.config.dynamic_hazard_density),
            self._entity_count(self.config.ppe_risk_density),
        )
        required = 1 + sum(counts)
        if required >= self.size * self.size:
            raise ValueError("Configured entity densities exceed available cells")
        indices = self.np_random.choice(self.size * self.size, size=required, replace=False)
        positions = [(int(index // self.size), int(index % self.size)) for index in indices]
        cursor = 0
        self.agent = positions[cursor]
        cursor += 1
        collections: list[set[Position]] = []
        for count in counts:
            collections.append(set(positions[cursor : cursor + count]))
            cursor += count
        (
            self.obstacles,
            self.hazards,
            self.workers,
            self.restricted,
            self.dynamic_hazards,
            self.ppe_risk,
        ) = collections
        self.orientation = int(self.np_random.integers(0, 4))

    def _visible(self, position: Position) -> bool:
        return (
            abs(position[0] - self.agent[0]) + abs(position[1] - self.agent[1])
            <= self.config.vision_radius
        )

    def _observed(self, position: Position) -> bool:
        if not self._visible(position):
            return False
        return bool(self.np_random.random() >= self.config.perception_false_negative_rate)

    def _observation(self) -> dict[str, np.ndarray]:
        grid = np.zeros((len(self.CHANNEL_NAMES), self.size, self.size), dtype=np.float32)
        for row in range(self.size):
            for column in range(self.size):
                if self._visible((row, column)):
                    grid[9, row, column] = 1.0
        layers = (
            (0, self.obstacles),
            (1, self.hazards - self.inspected),
            (2, self.workers),
            (3, self.restricted),
            (7, self.dynamic_hazards),
            (8, self.ppe_risk),
        )
        for channel, positions in layers:
            for position in sorted(positions):
                if self._observed(position):
                    grid[channel, position[0], position[1]] = 1.0
        for position in self.visited:
            grid[4, position[0], position[1]] = 1.0
        grid[5, self.agent[0], self.agent[1]] = 1.0
        risk_sources = (
            (self.hazards - self.inspected, 0.7),
            (self.workers, 0.9),
            (self.restricted, 1.0),
            (self.dynamic_hazards, 0.85),
            (self.ppe_risk, 0.65),
        )
        for positions, value in risk_sources:
            for position in positions:
                if self._visible(position):
                    grid[6, position[0], position[1]] = max(
                        grid[6, position[0], position[1]], value
                    )
        state = np.array(
            [
                self.agent[0] / max(1, self.size - 1),
                self.agent[1] / max(1, self.size - 1),
                self.orientation / 3.0,
                self.steps / self.config.max_steps,
                len(self.inspected) / max(1, len(self.hazards)),
                len(self.visited) / (self.size * self.size),
                min(1.0, self.cumulative_cost / 10.0),
                len(self.dynamic_hazards) / max(1, self.size * self.size),
                self.config.perception_false_negative_rate,
            ],
            dtype=np.float32,
        )
        return {"map": grid, "state": state}

    def _candidate(self, action: int) -> Position:
        delta = self.ACTION_TO_DELTA[action]
        return self.agent[0] + delta[0], self.agent[1] + delta[1]

    def _inside(self, position: Position) -> bool:
        return 0 <= position[0] < self.size and 0 <= position[1] < self.size

    def _near(self, position: Position, targets: set[Position], distance: int = 1) -> bool:
        return any(
            abs(position[0] - target[0]) + abs(position[1] - target[1]) <= distance
            for target in targets
        )

    def inspectable_hazards(self) -> set[Position]:
        return {
            hazard
            for hazard in self.hazards - self.inspected
            if abs(hazard[0] - self.agent[0]) + abs(hazard[1] - self.agent[1])
            <= self.config.inspection_radius
        }

    def action_safety_violations(self, action: int) -> tuple[str, ...]:
        if action == 4:
            position = self.agent
        elif action in self.ACTION_TO_DELTA:
            position = self._candidate(action)
            if not self._inside(position) or position in self.obstacles:
                return ("collision",)
        else:
            return ("invalid_action",)
        violations: list[str] = []
        if position in self.restricted:
            violations.append("restricted")
        if self._near(position, self.workers | self.dynamic_hazards):
            violations.append("near_miss")
        if position in self.ppe_risk:
            violations.append("ppe_risk")
        return tuple(violations)

    def action_masks(self) -> np.ndarray:
        mask = np.ones(self.action_space.n, dtype=np.bool_)
        for action in self.ACTION_TO_DELTA:
            candidate = self._candidate(action)
            mask[action] = self._inside(candidate) and candidate not in self.obstacles
        mask[4] = bool(self.inspectable_hazards())
        if not np.any(mask):
            mask[4] = True
        return mask

    def _advance_dynamic_hazards(self) -> None:
        if not self.config.dynamic_obstacles or not self.dynamic_hazards:
            return
        occupied = self.obstacles | self.workers | self.restricted | {self.agent}
        updated: set[Position] = set()
        for position in sorted(self.dynamic_hazards):
            candidates = [position]
            for delta in self.ACTION_TO_DELTA.values():
                candidate = position[0] + delta[0], position[1] + delta[1]
                if (
                    self._inside(candidate)
                    and candidate not in occupied
                    and candidate not in updated
                ):
                    candidates.append(candidate)
            updated.add(candidates[int(self.np_random.integers(0, len(candidates)))])
        self.dynamic_hazards = updated

    def step(self, action: int) -> tuple[dict[str, np.ndarray], float, bool, bool, dict[str, Any]]:
        if not self.action_space.contains(action):
            raise ValueError(f"Invalid action: {action}")
        self.steps += 1
        reward = -0.01
        cost = 0.0
        new_hazard = False
        if action in self.ACTION_TO_DELTA:
            self.orientation = action
            candidate = self._candidate(action)
            self.energy_usage += 1.0
            if not self._inside(candidate) or candidate in self.obstacles:
                reward -= 1.0
                cost += 1.0
                self.collisions += 1
            else:
                self.agent = candidate
                self.path_length += 1.0
                if candidate not in self.visited:
                    self.visited.add(candidate)
                    reward += 0.05
        else:
            self.energy_usage += 0.2
            inspectable = self.inspectable_hazards()
            if inspectable:
                self.inspected.update(inspectable)
                reward += 3.0 * len(inspectable)
                new_hazard = True
            else:
                reward -= 0.1
        if self.agent in self.restricted:
            cost += 1.0
            reward -= 0.75
            self.restricted_violations += 1
        if self._near(self.agent, self.workers | self.dynamic_hazards):
            cost += 1.0
            reward -= 0.5
            self.near_misses += 1
        if self.agent in self.ppe_risk:
            cost += 0.5
            reward -= 0.25
        self._advance_dynamic_hazards()
        terminated = len(self.inspected) == len(self.hazards)
        truncated = self.steps >= self.config.max_steps and not terminated
        if terminated:
            reward += 5.0
        self.cumulative_reward += reward
        self.cumulative_cost += cost
        return (
            self._observation(),
            float(reward),
            terminated,
            truncated,
            self._info(cost, new_hazard),
        )

    def _info(self, step_cost: float, new_hazard: bool) -> dict[str, Any]:
        return {
            "cost": float(step_cost),
            "safety_cost": float(step_cost),
            "new_hazard": bool(new_hazard),
            "hazards_inspected": len(self.inspected),
            "total_hazards": len(self.hazards),
            "hazard_recall": len(self.inspected) / max(1, len(self.hazards)),
            "coverage": len(self.visited) / (self.size * self.size),
            "collisions": self.collisions,
            "near_misses": self.near_misses,
            "restricted_violations": self.restricted_violations,
            "path_length": self.path_length,
            "energy_usage": self.energy_usage,
            "success": len(self.inspected) == len(self.hazards),
            "action_mask": self.action_masks().tolist(),
            "orientation": self.orientation,
        }

    def telemetry(self) -> EpisodeTelemetry:
        return EpisodeTelemetry(
            episode_seed=self.episode_seed,
            steps=self.steps,
            reward=self.cumulative_reward,
            safety_cost=self.cumulative_cost,
            hazards_inspected=len(self.inspected),
            total_hazards=len(self.hazards),
            collisions=self.collisions,
            near_misses=self.near_misses,
            restricted_violations=self.restricted_violations,
            path_length=self.path_length,
            energy_usage=self.energy_usage,
        )

    def serialize_state(self) -> dict[str, Any]:
        return {
            "config": asdict(self.config),
            "agent": list(self.agent),
            "orientation": self.orientation,
            "obstacles": sorted(self.obstacles),
            "hazards": sorted(self.hazards),
            "workers": sorted(self.workers),
            "restricted": sorted(self.restricted),
            "dynamic_hazards": sorted(self.dynamic_hazards),
            "ppe_risk": sorted(self.ppe_risk),
            "inspected": sorted(self.inspected),
            "visited": sorted(self.visited),
            "steps": self.steps,
            "telemetry": asdict(self.telemetry()),
        }

    def render(self) -> str | np.ndarray:
        canvas = np.full((self.size, self.size), " ", dtype="<U1")
        for positions, symbol in (
            (self.visited, "."),
            (self.obstacles, "#"),
            (self.restricted, "X"),
            (self.workers, "W"),
            (self.dynamic_hazards, "D"),
            (self.ppe_risk, "P"),
            (self.hazards - self.inspected, "H"),
            (self.inspected, "h"),
        ):
            for row, column in positions:
                canvas[row, column] = symbol
        canvas[self.agent[0], self.agent[1]] = "^v<>"[self.orientation]
        if self.render_mode == "rgb_array":
            palette = {
                " ": (245, 245, 245),
                ".": (210, 230, 245),
                "#": (40, 40, 40),
                "X": (220, 50, 50),
                "W": (255, 180, 40),
                "D": (170, 80, 200),
                "P": (255, 220, 30),
                "H": (230, 30, 30),
                "h": (60, 180, 75),
                "^": (30, 90, 220),
                "v": (30, 90, 220),
                "<": (30, 90, 220),
                ">": (30, 90, 220),
            }
            image = np.zeros((self.size, self.size, 3), dtype=np.uint8)
            for row in range(self.size):
                for column in range(self.size):
                    image[row, column] = palette[canvas[row, column]]
            return np.repeat(np.repeat(image, 16, axis=0), 16, axis=1)
        return "\n".join(" ".join(row) for row in canvas)
