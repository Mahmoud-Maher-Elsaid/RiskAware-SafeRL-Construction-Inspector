from __future__ import annotations

from riskaware_saferrl.envs import GridEnvironmentConfig, ResearchConstructionEnv
from riskaware_saferrl.safety import PredictiveSafetyShield


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
        )
    )
    env.reset(seed=2)
    env.agent = (4, 2)
    env.obstacles = set()
    env.workers = set()
    env.dynamic_hazards = set()
    env.restricted = set()
    env.ppe_risk = set()
    return env


def test_safe_action_is_accepted_deterministically() -> None:
    env = environment()
    shield = PredictiveSafetyShield(horizon=3)
    first = shield.decide(env, 3)
    second = shield.decide(env, 3)
    assert first.final_action == 3
    assert first.shield_decision == "accept"
    assert first.predicted_trajectory == second.predicted_trajectory
    assert first.computation_time_ms >= 0


def test_collision_and_restricted_boundary_are_predicted() -> None:
    env = environment()
    env.obstacles = {(4, 3)}
    env.restricted = {(3, 2)}
    decision = PredictiveSafetyShield(horizon=2).decide(env, 3)
    assert "collision" in decision.violation_types
    assert decision.final_action != 3


def test_moving_worker_near_miss_is_predicted() -> None:
    env = environment()
    env.dynamic_hazards = {(4, 4)}
    decision = PredictiveSafetyShield(horizon=3).decide(env, 3)
    assert "near_miss" in decision.violation_types
    assert decision.shield_decision in {"replace", "stop"}


def test_k_step_finds_risk_outside_one_step() -> None:
    env = environment()
    env.restricted = {(4, 4)}
    one_step = PredictiveSafetyShield(horizon=1).decide(env, 3)
    three_step = PredictiveSafetyShield(horizon=3).decide(env, 3)
    assert one_step.shield_decision == "accept"
    assert "restricted_zone" in three_step.violation_types


def test_no_safe_action_triggers_emergency_stop() -> None:
    env = environment()
    env.obstacles = {(3, 2), (5, 2), (4, 1), (4, 3)}
    env.restricted = {(4, 2)}
    decision = PredictiveSafetyShield(horizon=2).decide(env, 0)
    assert decision.emergency_stop
    assert decision.final_action == 4
