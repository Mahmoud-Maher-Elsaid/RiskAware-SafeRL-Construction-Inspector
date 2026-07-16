from riskaware_saferrl.envs.curriculum_env import (
    CurriculumConstructionInspectionEnv,
)
from riskaware_saferrl.scenarios import Scenario


def create_scenario(identifier: str, hazard: tuple[int, int]) -> Scenario:
    return Scenario(
        scenario_id=identifier,
        split="train",
        grid_size=6,
        agent_start=(0, 0),
        obstacles=(),
        hazards=(hazard,),
        workers=(),
        restricted_zones=(),
        max_steps=30,
        vision_radius=3,
    )


def create_environment() -> CurriculumConstructionInspectionEnv:
    return CurriculumConstructionInspectionEnv(
        {
            "easy": (create_scenario("easy_001", (0, 1)),),
            "medium": (create_scenario("medium_001", (2, 2)),),
            "hard": (create_scenario("hard_001", (5, 5)),),
        },
        inspection_radius=2,
    )


def test_curriculum_stages_expand_active_scenario_pool() -> None:
    environment = create_environment()

    assert environment.active_scenario_ids == ("easy_001",)

    medium_state = environment.set_curriculum_stage("medium")
    assert set(environment.active_scenario_ids) == {
        "easy_001",
        "medium_001",
    }
    assert medium_state["active_scenario_count"] == 2

    full_state = environment.set_curriculum_stage("full")
    assert set(environment.active_scenario_ids) == {
        "easy_001",
        "medium_001",
        "hard_001",
    }
    assert full_state["active_scenario_count"] == 3


def test_curriculum_reset_reports_stage_and_tier() -> None:
    environment = create_environment()
    _, info = environment.reset(seed=42)

    assert info["curriculum_stage"] == "easy"
    assert info["curriculum_scenario_tier"] == "easy"
    assert info["curriculum_active_scenarios"] == 1
