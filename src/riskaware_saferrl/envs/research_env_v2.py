from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from gymnasium import spaces

from riskaware_saferrl.envs.research_env import (
    GridEnvironmentConfig,
    Position,
    ResearchConstructionEnv,
)


@dataclass(frozen=True)
class RewardCostV2Config:
    """Task reward and safety-cost scales for the v2 research environment."""

    gamma: float = 0.99
    step_penalty: float = 0.01
    exploration_reward: float = 0.08
    inspection_reward: float = 4.0
    completion_reward: float = 12.0
    potential_scale: float = 0.25
    useless_inspection_penalty: float = 0.12
    repeated_action_penalty: float = 0.015
    deadlock_penalty: float = 0.08
    timeout_penalty: float = 2.0
    collision_cost: float = 1.0
    near_miss_cost: float = 0.5
    restricted_cost: float = 1.0
    ppe_risk_cost: float = 0.25
    safety_budget: float = 20.0

    def validate(self) -> None:
        if not 0.0 < self.gamma <= 1.0:
            raise ValueError("gamma must be in (0, 1]")
        if self.safety_budget <= 0.0:
            raise ValueError("safety_budget must be positive")


class ResearchConstructionEnvV2(ResearchConstructionEnv):
    """Memory-ready research task with persistent semantics and separate costs.

    The original environment is intentionally unchanged for baseline
    reproducibility. V2 retains the same five actions and exact completion
    condition while fixing observation persistence and objective scaling.
    """

    V2_CHANNEL_NAMES = ResearchConstructionEnv.CHANNEL_NAMES + ("inspected_targets",)
    STATE_NAMES = (
        "row",
        "column",
        "orientation_sin",
        "orientation_cos",
        "time_fraction",
        "inspection_progress",
        "explored_fraction",
        "remaining_safety_budget",
        "perception_false_negative_rate",
        "map_size_fraction",
        "previous_action_north",
        "previous_action_south",
        "previous_action_west",
        "previous_action_east",
        "previous_action_inspect",
        "previous_reward",
        "previous_safety_cost",
    )

    def __init__(
        self,
        config: GridEnvironmentConfig | None = None,
        *,
        reward_cost_config: RewardCostV2Config | None = None,
        render_mode: str | None = None,
    ) -> None:
        super().__init__(config, render_mode=render_mode)
        self.reward_cost_config = reward_cost_config or RewardCostV2Config()
        self.reward_cost_config.validate()
        self.observation_space = spaces.Dict(
            {
                "map": spaces.Box(
                    0.0,
                    1.0,
                    (
                        len(self.V2_CHANNEL_NAMES),
                        self.config.observation_size,
                        self.config.observation_size,
                    ),
                    dtype=np.float32,
                ),
                "state": spaces.Box(-1.0, 1.0, (len(self.STATE_NAMES),), dtype=np.float32),
                "action_mask": spaces.MultiBinary(int(self.action_space.n)),
            }
        )
        self._semantic_memory = np.zeros(
            (
                len(self.V2_CHANNEL_NAMES),
                self.config.observation_size,
                self.config.observation_size,
            ),
            dtype=np.float32,
        )
        self.previous_action: int | None = None
        self.previous_reward = 0.0
        self.previous_cost = 0.0
        self._recent_positions: list[Position] = []
        self._detection_state: dict[tuple[int, Position], bool] = {}

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        # Invoke Gymnasium's seed initialization directly: the v1 reset creates
        # an observation before v2 episode memory can be cleared.
        super(ResearchConstructionEnv, self).reset(seed=seed)
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
        self._semantic_memory.fill(0.0)
        self.previous_action = None
        self.previous_reward = 0.0
        self.previous_cost = 0.0
        self._recent_positions = [self.agent]
        self._detection_state = {}
        observation = self._observation()
        return observation, self._info(0.0, False)

    def _detected_this_episode(self, channel: int, position: Position) -> bool:
        key = (channel, position)
        if key not in self._detection_state:
            self._detection_state[key] = bool(
                self.np_random.random() >= self.config.perception_false_negative_rate
            )
        return self._detection_state[key]

    def _observation(self) -> dict[str, np.ndarray]:
        current = np.zeros_like(self._semantic_memory)
        for row in range(self.size):
            for column in range(self.size):
                if self._visible((row, column)):
                    current[9, row, column] = 1.0
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
                if self._visible(position) and self._detected_this_episode(channel, position):
                    current[channel, position[0], position[1]] = 1.0
        for position in self.visited:
            current[4, position[0], position[1]] = 1.0
        current[5, self.agent[0], self.agent[1]] = 1.0
        for position in self.inspected:
            current[10, position[0], position[1]] = 1.0
        for channel, value in ((1, 0.7), (2, 0.9), (3, 1.0), (7, 0.85), (8, 0.65)):
            current[6] = np.maximum(current[6], current[channel] * value)
        # Static semantic evidence persists. Dynamic, agent, and visibility
        # channels represent the current frame.
        for channel in (0, 1, 3, 4, 8, 10):
            self._semantic_memory[channel] = np.maximum(
                self._semantic_memory[channel], current[channel]
            )
        for channel in (2, 5, 6, 7, 9):
            self._semantic_memory[channel] = current[channel]
        angle = self.orientation * np.pi / 2.0
        previous_action = np.zeros(5, dtype=np.float32)
        if self.previous_action is not None:
            previous_action[self.previous_action] = 1.0
        state = np.array(
            [
                self.agent[0] / max(1, self.size - 1),
                self.agent[1] / max(1, self.size - 1),
                np.sin(angle),
                np.cos(angle),
                self.steps / self.config.max_steps,
                len(self.inspected) / max(1, len(self.hazards)),
                len(self.visited) / (self.size * self.size),
                max(
                    -1.0,
                    1.0 - self.cumulative_cost / self.reward_cost_config.safety_budget,
                ),
                self.config.perception_false_negative_rate,
                self.size / self.config.observation_size,
                *previous_action,
                np.tanh(self.previous_reward / 5.0),
                np.tanh(self.previous_cost),
            ],
            dtype=np.float32,
        )
        return {
            "map": self._semantic_memory.copy(),
            "state": state,
            "action_mask": self.action_masks().astype(np.int8),
        }

    def _potential(self) -> float:
        remaining = self.hazards - self.inspected
        if not remaining:
            return 0.0
        distance = min(
            abs(self.agent[0] - row) + abs(self.agent[1] - column) for row, column in remaining
        )
        return -float(distance) / max(1.0, 2.0 * (self.size - 1))

    def step(self, action: int) -> tuple[dict[str, np.ndarray], float, bool, bool, dict[str, Any]]:
        if not self.action_space.contains(action):
            raise ValueError(f"Invalid action: {action}")
        cfg = self.reward_cost_config
        old_potential = self._potential()
        old_agent = self.agent
        old_inspected = len(self.inspected)
        self.steps += 1
        reward = -cfg.step_penalty
        cost = 0.0
        new_hazard = False
        if action in self.ACTION_TO_DELTA:
            self.orientation = action
            candidate = self._candidate(action)
            self.energy_usage += 1.0
            if not self._inside(candidate) or candidate in self.obstacles:
                cost += cfg.collision_cost
                self.collisions += 1
            else:
                self.agent = candidate
                self.path_length += 1.0
                if candidate not in self.visited:
                    self.visited.add(candidate)
                    reward += cfg.exploration_reward
        else:
            self.energy_usage += 0.2
            inspectable = self.inspectable_hazards()
            if inspectable:
                self.inspected.update(inspectable)
                reward += cfg.inspection_reward * len(inspectable)
                new_hazard = True
            else:
                reward -= cfg.useless_inspection_penalty
        if self.agent in self.restricted:
            cost += cfg.restricted_cost
            self.restricted_violations += 1
        if self._near(self.agent, self.workers | self.dynamic_hazards):
            cost += cfg.near_miss_cost
            self.near_misses += 1
        if self.agent in self.ppe_risk:
            cost += cfg.ppe_risk_cost
        self._advance_dynamic_hazards()
        reward += cfg.potential_scale * (cfg.gamma * self._potential() - old_potential)
        if self.previous_action == action:
            reward -= cfg.repeated_action_penalty
        self._recent_positions.append(self.agent)
        self._recent_positions = self._recent_positions[-8:]
        if (
            len(self._recent_positions) == 8
            and len(set(self._recent_positions)) <= 2
            and len(self.inspected) == old_inspected
        ):
            reward -= cfg.deadlock_penalty
        terminated = len(self.inspected) == len(self.hazards)
        truncated = self.steps >= self.config.max_steps and not terminated
        if terminated:
            reward += cfg.completion_reward
        elif truncated:
            reward -= cfg.timeout_penalty
        self.cumulative_reward += reward
        self.cumulative_cost += cost
        self.previous_action = action
        self.previous_reward = float(reward)
        self.previous_cost = float(cost)
        observation = self._observation()
        info = self._info(cost, new_hazard)
        info.update(
            {
                "inspection_coverage": len(self.inspected) / max(1, len(self.hazards)),
                "explored_area_coverage": len(self.visited) / (self.size * self.size),
                "remaining_safety_budget": cfg.safety_budget - self.cumulative_cost,
                "moved": self.agent != old_agent,
                "reward_version": "v2",
            }
        )
        return observation, float(reward), terminated, truncated, info
