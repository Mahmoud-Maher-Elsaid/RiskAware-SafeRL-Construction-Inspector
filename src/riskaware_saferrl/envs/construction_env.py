from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces


class ConstructionInspectionEnv(gym.Env):
    """Partially observable construction-site inspection environment."""

    metadata = {"render_modes": ["ansi"], "render_fps": 4}

    ACTION_TO_DELTA = {
        0: np.array([-1, 0], dtype=np.int32),
        1: np.array([1, 0], dtype=np.int32),
        2: np.array([0, -1], dtype=np.int32),
        3: np.array([0, 1], dtype=np.int32),
    }

    def __init__(
        self,
        size: int = 12,
        n_obstacles: int = 18,
        n_hazards: int = 6,
        n_workers: int = 4,
        n_restricted: int = 8,
        max_steps: int = 250,
        vision_radius: int = 3,
        render_mode: str | None = None,
    ) -> None:
        super().__init__()

        if size < 6:
            raise ValueError("size must be at least 6")

        requested_cells = 1 + n_obstacles + n_hazards + n_workers + n_restricted
        if requested_cells >= size * size:
            raise ValueError("Too many entities for the selected grid size")

        self.size = size
        self.n_obstacles = n_obstacles
        self.n_hazards = n_hazards
        self.n_workers = n_workers
        self.n_restricted = n_restricted
        self.max_steps = max_steps
        self.vision_radius = vision_radius
        self.render_mode = render_mode

        self.action_space = spaces.Discrete(5)
        self.observation_space = spaces.Dict(
            {
                "map": spaces.Box(
                    low=0.0,
                    high=1.0,
                    shape=(7, size, size),
                    dtype=np.float32,
                ),
                "state": spaces.Box(
                    low=0.0,
                    high=1.0,
                    shape=(4,),
                    dtype=np.float32,
                ),
            }
        )

        self.agent = np.zeros(2, dtype=np.int32)
        self.obstacles: set[tuple[int, int]] = set()
        self.hazards: set[tuple[int, int]] = set()
        self.workers: set[tuple[int, int]] = set()
        self.restricted: set[tuple[int, int]] = set()
        self.inspected: set[tuple[int, int]] = set()
        self.visited: set[tuple[int, int]] = set()
        self.steps = 0

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        super().reset(seed=seed)

        positions = np.array(
            [(row, col) for row in range(self.size) for col in range(self.size)],
            dtype=np.int32,
        )
        count = 1 + self.n_obstacles + self.n_hazards + self.n_workers + self.n_restricted
        selected = positions[self.np_random.choice(len(positions), size=count, replace=False)]

        index = 0
        self.agent = selected[index].copy()
        index += 1

        self.obstacles = self._to_position_set(selected[index : index + self.n_obstacles])
        index += self.n_obstacles

        self.hazards = self._to_position_set(selected[index : index + self.n_hazards])
        index += self.n_hazards

        self.workers = self._to_position_set(selected[index : index + self.n_workers])
        index += self.n_workers

        self.restricted = self._to_position_set(selected[index : index + self.n_restricted])

        self.inspected = set()
        self.visited = {self.agent_position}
        self.steps = 0

        info = self._get_info(
            collision_cost=0.0,
            worker_cost=0.0,
            restricted_cost=0.0,
            new_hazard=False,
        )
        return self._get_obs(), info

    @staticmethod
    def _to_position_set(values: np.ndarray) -> set[tuple[int, int]]:
        return {(int(row), int(col)) for row, col in values}

    @property
    def agent_position(self) -> tuple[int, int]:
        return int(self.agent[0]), int(self.agent[1])

    def _is_visible(self, position: tuple[int, int]) -> bool:
        return (
            abs(position[0] - self.agent_position[0])
            + abs(position[1] - self.agent_position[1])
            <= self.vision_radius
        )

    def _get_obs(self) -> dict[str, np.ndarray]:
        grid = np.zeros((7, self.size, self.size), dtype=np.float32)

        for position in self.obstacles:
            if self._is_visible(position):
                grid[0, position[0], position[1]] = 1.0

        for position in self.hazards - self.inspected:
            if self._is_visible(position):
                grid[1, position[0], position[1]] = 1.0
                grid[6, position[0], position[1]] = max(
                    grid[6, position[0], position[1]], 0.7
                )

        for position in self.workers:
            if self._is_visible(position):
                grid[2, position[0], position[1]] = 1.0
                self._paint_local_risk(grid[6], position, 0.85)

        for position in self.restricted:
            if self._is_visible(position):
                grid[3, position[0], position[1]] = 1.0
                grid[6, position[0], position[1]] = 1.0

        for position in self.visited:
            grid[4, position[0], position[1]] = 1.0

        grid[5, self.agent_position[0], self.agent_position[1]] = 1.0

        inspected_ratio = len(self.inspected) / max(1, self.n_hazards)
        state = np.array(
            [
                self.agent[0] / max(1, self.size - 1),
                self.agent[1] / max(1, self.size - 1),
                self.steps / max(1, self.max_steps),
                inspected_ratio,
            ],
            dtype=np.float32,
        )

        return {"map": grid, "state": state}

    def _paint_local_risk(
        self,
        risk_map: np.ndarray,
        center: tuple[int, int],
        value: float,
    ) -> None:
        for row_delta, col_delta in ((0, 0), (-1, 0), (1, 0), (0, -1), (0, 1)):
            row = center[0] + row_delta
            col = center[1] + col_delta
            if 0 <= row < self.size and 0 <= col < self.size:
                risk_map[row, col] = max(risk_map[row, col], value)

    def is_action_safe(self, action: int) -> bool:
        if action == 4:
            return True

        if action not in self.ACTION_TO_DELTA:
            return False

        candidate = self.agent + self.ACTION_TO_DELTA[action]
        candidate_position = int(candidate[0]), int(candidate[1])

        if not self._inside(candidate):
            return False
        if candidate_position in self.obstacles:
            return False
        if candidate_position in self.restricted:
            return False
        if self._near_worker(candidate_position):
            return False

        return True

    def safe_actions(self) -> list[int]:
        return [action for action in range(self.action_space.n) if self.is_action_safe(action)]

    def step(
        self,
        action: int,
    ) -> tuple[dict[str, np.ndarray], float, bool, bool, dict[str, Any]]:
        if not self.action_space.contains(action):
            raise ValueError(f"Invalid action: {action}")

        self.steps += 1
        reward = -0.01
        collision_cost = 0.0
        worker_cost = 0.0
        restricted_cost = 0.0
        new_hazard = False

        if action in self.ACTION_TO_DELTA:
            candidate = self.agent + self.ACTION_TO_DELTA[action]
            candidate_position = int(candidate[0]), int(candidate[1])

            if not self._inside(candidate) or candidate_position in self.obstacles:
                collision_cost = 1.0
                reward -= 1.0
            else:
                self.agent = candidate

                if self.agent_position not in self.visited:
                    reward += 0.05
                    self.visited.add(self.agent_position)

                if self.agent_position in self.restricted:
                    restricted_cost = 1.0
                    reward -= 0.75

                if self._near_worker(self.agent_position):
                    worker_cost = 1.0
                    reward -= 0.5

        elif action == 4:
            if self.agent_position in self.hazards and self.agent_position not in self.inspected:
                self.inspected.add(self.agent_position)
                reward += 3.0
                new_hazard = True
            else:
                reward -= 0.1

            if self._near_worker(self.agent_position):
                worker_cost = 1.0
                reward -= 0.5

        terminated = len(self.inspected) == self.n_hazards
        truncated = self.steps >= self.max_steps and not terminated

        if terminated:
            reward += 5.0

        info = self._get_info(
            collision_cost=collision_cost,
            worker_cost=worker_cost,
            restricted_cost=restricted_cost,
            new_hazard=new_hazard,
        )
        return self._get_obs(), float(reward), terminated, truncated, info

    def _inside(self, position: np.ndarray) -> bool:
        return bool(np.all(position >= 0) and np.all(position < self.size))

    def _near_worker(self, position: tuple[int, int]) -> bool:
        return any(
            abs(position[0] - worker[0]) + abs(position[1] - worker[1]) <= 1
            for worker in self.workers
        )

    def _get_info(
        self,
        *,
        collision_cost: float,
        worker_cost: float,
        restricted_cost: float,
        new_hazard: bool,
    ) -> dict[str, Any]:
        total_cost = collision_cost + worker_cost + restricted_cost
        return {
            "cost": float(total_cost),
            "cost_collision": float(collision_cost),
            "cost_worker": float(worker_cost),
            "cost_restricted": float(restricted_cost),
            "new_hazard": bool(new_hazard),
            "hazards_inspected": len(self.inspected),
            "total_hazards": self.n_hazards,
            "hazard_recall": len(self.inspected) / max(1, self.n_hazards),
            "coverage": len(self.visited) / (self.size * self.size),
            "success": len(self.inspected) == self.n_hazards,
        }

    def render(self) -> str:
        canvas = np.full((self.size, self.size), " ", dtype="<U1")

        for row, col in self.visited:
            canvas[row, col] = "."

        for row, col in self.obstacles:
            canvas[row, col] = "#"

        for row, col in self.restricted:
            canvas[row, col] = "X"

        for row, col in self.workers:
            canvas[row, col] = "W"

        for row, col in self.hazards:
            canvas[row, col] = "h" if (row, col) in self.inspected else "H"

        canvas[self.agent_position[0], self.agent_position[1]] = "A"
        return "\n".join(" ".join(row) for row in canvas)

    def close(self) -> None:
        return None