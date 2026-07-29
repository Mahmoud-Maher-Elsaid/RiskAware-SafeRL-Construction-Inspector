from __future__ import annotations

import pytest

from riskaware_saferrl.baselines import (
    FrontierExplorationPlanner,
    NearestRiskRevisitPlanner,
    RiskAwareAStarPlanner,
)
from riskaware_saferrl.envs import GridEnvironmentConfig, ResearchConstructionEnv


@pytest.fixture
def environment() -> ResearchConstructionEnv:
    env = ResearchConstructionEnv(
        GridEnvironmentConfig(
            size=8,
            obstacle_density=0,
            hazard_density=0.02,
            worker_density=0,
            restricted_density=0,
            dynamic_hazard_density=0,
            ppe_risk_density=0,
            dynamic_obstacles=False,
            max_steps=60,
        )
    )
    env.reset(seed=42)
    env.agent = (4, 1)
    env.hazards = {(4, 5)}
    env.obstacles = {(4, 3)}
    env.restricted = {(3, 2)}
    env.workers = {(5, 2)}
    env.dynamic_hazards = set()
    env.ppe_risk = set()
    env.inspected = set()
    env.visited = {env.agent}
    return env


@pytest.mark.parametrize(
    "planner_type",
    [RiskAwareAStarPlanner, FrontierExplorationPlanner, NearestRiskRevisitPlanner],
)
def test_planners_execute_valid_deterministic_action(environment, planner_type) -> None:
    first = planner_type()
    second = planner_type()
    first_decision = first.decide(environment)
    second_decision = second.decide(environment)
    assert first_decision == second_decision
    assert environment.action_space.contains(first_decision.action)


def test_astar_avoids_restricted_and_obstacle(environment) -> None:
    planner = RiskAwareAStarPlanner(risk_weight=10.0)
    decision = planner.decide(environment)
    assert (4, 3) not in decision.path
    assert (3, 2) not in decision.path


def test_inspection_action_support(environment) -> None:
    environment.agent = (4, 4)
    for planner_type in (
        RiskAwareAStarPlanner,
        FrontierExplorationPlanner,
        NearestRiskRevisitPlanner,
    ):
        assert planner_type().decide(environment).action == 4


def test_unreachable_target_returns_hold(environment) -> None:
    environment.obstacles = {
        (row, column)
        for row in range(environment.size)
        for column in range(environment.size)
        if (row, column) != environment.agent
    }
    decision = RiskAwareAStarPlanner().decide(environment)
    assert decision.action == 4
    assert decision.reason == "no_reachable_target"


def test_dynamic_obstacle_replanning_changes_path(environment) -> None:
    planner = RiskAwareAStarPlanner()
    first = planner.decide(environment)
    if len(first.path) > 1:
        environment.dynamic_hazards = {first.path[1]}
    second = planner.decide(environment)
    assert second.replanned
    assert second.path != first.path or len(first.path) == 1
