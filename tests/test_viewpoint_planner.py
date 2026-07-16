from riskaware_saferrl.evaluation.expert_baselines import (
    evaluate_plan,
)
from riskaware_saferrl.planners import (
    build_viewpoint_inspection_plan,
)
from riskaware_saferrl.scenarios import Scenario


def create_worker_blocked_hazard_scenario() -> Scenario:
    return Scenario(
        scenario_id="safe_viewpoint_000001",
        split="test",
        grid_size=6,
        agent_start=(0, 0),
        obstacles=(),
        hazards=((2, 2),),
        workers=((2, 3),),
        restricted_zones=(),
        max_steps=50,
        vision_radius=3,
    )


def test_safe_viewpoint_plan_avoids_unsafe_hazard_cell() -> None:
    scenario = create_worker_blocked_hazard_scenario()

    plan = build_viewpoint_inspection_plan(
        scenario,
        safety_aware=True,
        inspection_radius=2,
    )

    assert plan.complete
    assert (2, 2) not in plan.viewpoints

    record = evaluate_plan(
        scenario,
        plan,
    )

    assert record["success"] == 1.0
    assert record["safety_cost"] == 0.0


def test_safe_viewpoint_plan_is_deterministic() -> None:
    scenario = create_worker_blocked_hazard_scenario()

    first = build_viewpoint_inspection_plan(
        scenario,
        safety_aware=True,
        inspection_radius=2,
    )
    second = build_viewpoint_inspection_plan(
        scenario,
        safety_aware=True,
        inspection_radius=2,
    )

    assert first == second
