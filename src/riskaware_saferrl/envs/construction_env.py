from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from riskaware_saferrl.inspection import inspectable_hazards_from
from riskaware_saferrl.scenarios import Scenario


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
        inspection_radius: int = 2,
        render_mode: str | None = None,
        scenario: Scenario | None = None,
        scenarios: Sequence[Scenario] | None = None,
    ) -> None:
        super().__init__()

        if scenario is not None and scenarios is not None:
            raise ValueError("Provide either scenario or scenarios, not both.")

        self.fixed_scenario = scenario
        self.scenario_pool = tuple(scenarios) if scenarios is not None else ()
        self.current_scenario: Scenario | None = None

        configured_scenarios = (scenario,) if scenario is not None else self.scenario_pool

        if configured_scenarios:
            for configured_scenario in configured_scenarios:
                configured_scenario.validate()

            scenario_sizes = {
                configured_scenario.grid_size for configured_scenario in configured_scenarios
            }
            if len(scenario_sizes) != 1:
                raise ValueError("All scenarios in one environment must use the same grid size.")

            size = configured_scenarios[0].grid_size

        if size < 6:
            raise ValueError("size must be at least 6")
        if inspection_radius < 0:
            raise ValueError("inspection_radius must be non-negative.")

        requested_cells = 1 + n_obstacles + n_hazards + n_workers + n_restricted
        if not configured_scenarios and requested_cells >= size * size:
            raise ValueError("Too many entities for the selected grid size")

        self.size = size
        self.n_obstacles = n_obstacles
        self.n_hazards = n_hazards
        self.n_workers = n_workers
        self.n_restricted = n_restricted
        self.max_steps = max_steps
        self.vision_radius = vision_radius
        self.inspection_radius = inspection_radius
        self.render_mode = render_mode

        self.action_space = spaces.Discrete(5)
        self.observation_space = spaces.Dict(
            {
                "map": spaces.Box(
                    low=0.0,
                    high=1.0,
                    shape=(7 * size * size,),
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

        selected_scenario = self._select_scenario(options)

        if selected_scenario is None:
            self._generate_random_layout()
        else:
            self._load_scenario(selected_scenario)

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

    def _select_scenario(self, options: dict[str, Any] | None) -> Scenario | None:
        if self.fixed_scenario is not None:
            return self.fixed_scenario

        if not self.scenario_pool:
            return None

        if options is not None and "scenario_index" in options:
            scenario_index = int(options["scenario_index"])
            if not 0 <= scenario_index < len(self.scenario_pool):
                raise IndexError(f"Scenario index is out of range: {scenario_index}")
            return self.scenario_pool[scenario_index]

        scenario_index = int(self.np_random.integers(0, len(self.scenario_pool)))
        return self.scenario_pool[scenario_index]

    def _load_scenario(self, scenario: Scenario) -> None:
        if scenario.grid_size != self.size:
            raise ValueError(
                f"Scenario grid size {scenario.grid_size} does not match "
                f"environment grid size {self.size}."
            )

        self.current_scenario = scenario
        self.agent = np.array(scenario.agent_start, dtype=np.int32)
        self.obstacles = set(scenario.obstacles)
        self.hazards = set(scenario.hazards)
        self.workers = set(scenario.workers)
        self.restricted = set(scenario.restricted_zones)

        self.n_obstacles = len(self.obstacles)
        self.n_hazards = len(self.hazards)
        self.n_workers = len(self.workers)
        self.n_restricted = len(self.restricted)
        self.max_steps = scenario.max_steps
        self.vision_radius = scenario.vision_radius

    def _generate_random_layout(self) -> None:
        self.current_scenario = None

        positions = np.array(
            [(row, col) for row in range(self.size) for col in range(self.size)],
            dtype=np.int32,
        )
        count = 1 + self.n_obstacles + self.n_hazards + self.n_workers + self.n_restricted
        selected = positions[
            self.np_random.choice(
                len(positions),
                size=count,
                replace=False,
            )
        ]

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

    @staticmethod
    def _to_position_set(values: np.ndarray) -> set[tuple[int, int]]:
        return {(int(row), int(col)) for row, col in values}

    @property
    def agent_position(self) -> tuple[int, int]:
        return int(self.agent[0]), int(self.agent[1])

    def _is_visible(self, position: tuple[int, int]) -> bool:
        return (
            abs(position[0] - self.agent_position[0]) + abs(position[1] - self.agent_position[1])
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
                    grid[6, position[0], position[1]],
                    0.7,
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

        return {"map": grid.reshape(-1), "state": state}

    def semantic_grid_from_observation(
        self,
        observation: dict[str, np.ndarray],
    ) -> np.ndarray:
        """Restore the flattened semantic map for plotting and debugging."""
        return observation["map"].reshape(7, self.size, self.size)

    def _paint_local_risk(
        self,
        risk_map: np.ndarray,
        center: tuple[int, int],
        value: float,
    ) -> None:
        for row_delta, col_delta in (
            (0, 0),
            (-1, 0),
            (1, 0),
            (0, -1),
            (0, 1),
        ):
            row = center[0] + row_delta
            col = center[1] + col_delta
            if 0 <= row < self.size and 0 <= col < self.size:
                risk_map[row, col] = max(risk_map[row, col], value)

    def inspectable_hazards(
        self,
        position: tuple[int, int] | None = None,
        *,
        include_inspected: bool = False,
    ) -> frozenset[tuple[int, int]]:
        viewpoint = self.agent_position if position is None else position
        hazards = self.hazards

        if not include_inspected:
            hazards = hazards - self.inspected

        return inspectable_hazards_from(
            viewpoint,
            hazards,
            blockers=self.obstacles,
            inspection_radius=self.inspection_radius,
        )

    def action_safety_violations(self, action: int) -> tuple[str, ...]:
        if action == 4:
            return ()

        if action not in self.ACTION_TO_DELTA:
            return ("invalid_action",)

        candidate = self.agent + self.ACTION_TO_DELTA[action]
        candidate_position = int(candidate[0]), int(candidate[1])
        violations: list[str] = []

        if not self._inside(candidate) or candidate_position in self.obstacles:
            violations.append("collision")
            return tuple(violations)

        if candidate_position in self.restricted:
            violations.append("restricted")

        if self._near_worker(candidate_position):
            violations.append("worker")

        return tuple(violations)

    def is_action_safe(self, action: int) -> bool:
        return not self.action_safety_violations(action)

    def safe_actions(self) -> list[int]:
        return [action for action in range(self.action_space.n) if self.is_action_safe(action)]

    def action_masks(self) -> np.ndarray:
        masks = np.zeros(self.action_space.n, dtype=np.bool_)

        for action, delta in self.ACTION_TO_DELTA.items():
            candidate = self.agent + delta
            candidate_position = int(candidate[0]), int(candidate[1])
            masks[action] = self._inside(candidate) and candidate_position not in self.obstacles

        masks[4] = bool(self.inspectable_hazards())

        if not bool(np.any(masks)):
            masks[4] = True

        return masks

    def task_valid_actions(self) -> list[int]:
        return [action for action, is_valid in enumerate(self.action_masks()) if bool(is_valid)]

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
        new_hazard_count = 0

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
            newly_inspected = self.inspectable_hazards()

            if newly_inspected:
                self.inspected.update(newly_inspected)
                new_hazard_count = len(newly_inspected)
                reward += 3.0 * new_hazard_count
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
            new_hazard_count=new_hazard_count,
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
        new_hazard_count: int = 0,
    ) -> dict[str, Any]:
        total_cost = collision_cost + worker_cost + restricted_cost

        return {
            "cost": float(total_cost),
            "cost_collision": float(collision_cost),
            "cost_worker": float(worker_cost),
            "cost_restricted": float(restricted_cost),
            "new_hazard": bool(new_hazard),
            "new_hazards": int(new_hazard_count),
            "inspectable_hazards": len(self.inspectable_hazards()),
            "inspection_radius": self.inspection_radius,
            "hazards_inspected": len(self.inspected),
            "total_hazards": self.n_hazards,
            "hazard_recall": len(self.inspected) / max(1, self.n_hazards),
            "coverage": len(self.visited) / (self.size * self.size),
            "success": len(self.inspected) == self.n_hazards,
            "scenario_id": (
                self.current_scenario.scenario_id if self.current_scenario is not None else None
            ),
            "scenario_split": (
                self.current_scenario.split if self.current_scenario is not None else None
            ),
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
