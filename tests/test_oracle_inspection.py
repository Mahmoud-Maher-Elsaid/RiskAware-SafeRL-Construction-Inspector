from riskaware_saferrl.planners import (
    build_oracle_inspection_plan,
)
from riskaware_saferrl.scenarios import Scenario


def create_simple_scenario() -> Scenario:
    return Scenario(
        scenario_id="simple_000001",
        split="test",
        grid_size=6,
        agent_start=(0, 0),
        obstacles=((1, 1),),
        hazards=((0, 2), (2, 2)),
        workers=((5, 5),),
        restricted_zones=((4, 4),),
        max_steps=100,
        vision_radius=3,
    )


def test_oracle_plan_visits_all_hazards() -> None:
    scenario = create_simple_scenario()

    plan = build_oracle_inspection_plan(
        scenario,
        safety_aware=False,
    )

    assert plan.complete
    assert set(plan.visited_hazards) == set(scenario.hazards)
    assert plan.inspection_actions == 2


def test_safe_oracle_is_deterministic() -> None:
    scenario = create_simple_scenario()

    first = build_oracle_inspection_plan(
        scenario,
        safety_aware=True,
    )
    second = build_oracle_inspection_plan(
        scenario,
        safety_aware=True,
    )

    assert first == second
