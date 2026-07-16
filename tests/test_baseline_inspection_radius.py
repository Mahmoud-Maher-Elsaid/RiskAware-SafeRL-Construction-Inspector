from riskaware_saferrl.evaluation.expert_baselines import (
    evaluate_all_baselines,
    evaluate_inspect_only,
)
from riskaware_saferrl.scenarios import Scenario


def create_radius_scenario() -> Scenario:
    return Scenario(
        scenario_id="radius_000001",
        split="test",
        grid_size=6,
        agent_start=(0, 0),
        obstacles=(),
        hazards=((0, 3),),
        workers=(),
        restricted_zones=(),
        max_steps=20,
        vision_radius=3,
    )


def test_inspect_only_uses_requested_inspection_radius() -> None:
    scenario = create_radius_scenario()

    radius_one = evaluate_inspect_only(
        scenario,
        inspection_radius=1,
    )
    radius_three = evaluate_inspect_only(
        scenario,
        inspection_radius=3,
    )

    assert radius_one["success"] == 0.0
    assert radius_three["success"] == 1.0
    assert radius_one["inspection_radius"] == 1.0
    assert radius_three["inspection_radius"] == 3.0


def test_all_baselines_propagate_inspection_radius() -> None:
    records = evaluate_all_baselines(
        [create_radius_scenario()],
        random_seed=42,
        inspection_radius=3,
    )

    inspect_record = next(record for record in records if record["planner"] == "inspect_only")
    random_record = next(record for record in records if record["planner"] == "random")

    assert inspect_record["inspection_radius"] == 3.0
    assert random_record["inspection_radius"] == 3.0
