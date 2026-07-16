import numpy as np
from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.utils import get_action_masks
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from riskaware_saferrl.envs import ConstructionInspectionEnv
from riskaware_saferrl.safety import SafetyShield
from riskaware_saferrl.scenarios import Scenario


def create_scenario(
    *,
    scenario_id: str,
    hazards: tuple[tuple[int, int], ...] = ((0, 0),),
    workers: tuple[tuple[int, int], ...] = (),
    restricted_zones: tuple[tuple[int, int], ...] = (),
) -> Scenario:
    return Scenario(
        scenario_id=scenario_id,
        split="test",
        grid_size=6,
        agent_start=(2, 2),
        obstacles=(),
        hazards=hazards,
        workers=workers,
        restricted_zones=restricted_zones,
        max_steps=20,
        vision_radius=3,
    )


def test_environment_reports_worker_and_restricted_violations() -> None:
    environment = ConstructionInspectionEnv(
        scenario=create_scenario(
            scenario_id="combined_violation",
            workers=((2, 4),),
            restricted_zones=((2, 3),),
        )
    )
    environment.reset(seed=0)

    violations = environment.action_safety_violations(3)

    assert violations == ("restricted", "worker")
    assert not environment.is_action_safe(3)


def test_shield_preserves_a_safe_action() -> None:
    environment = SafetyShield(
        ConstructionInspectionEnv(
            scenario=create_scenario(
                scenario_id="safe_action",
                workers=((2, 4),),
            )
        )
    )
    environment.reset(seed=0)

    _, _, _, _, info = environment.step(0)

    assert info["shield_active"] is False
    assert info["proposed_action"] == 0
    assert info["executed_action"] == 0


def test_shield_projects_worker_risk_to_minimum_deviation() -> None:
    environment = SafetyShield(
        ConstructionInspectionEnv(
            scenario=create_scenario(
                scenario_id="worker_projection",
                workers=((2, 4),),
            )
        )
    )
    environment.reset(seed=0)

    _, _, _, _, info = environment.step(3)

    assert info["shield_active"] is True
    assert "worker" in info["shield_violations"]
    assert info["executed_action"] == 0
    assert info["shield_replacement_safe"] is True
    assert info["shield_replacement_task_valid"] is True
    assert info["cost"] == 0.0


def test_shield_projects_restricted_action() -> None:
    environment = SafetyShield(
        ConstructionInspectionEnv(
            scenario=create_scenario(
                scenario_id="restricted_projection",
                restricted_zones=((3, 2),),
            )
        )
    )
    environment.reset(seed=0)

    _, _, _, _, info = environment.step(1)

    assert info["shield_active"] is True
    assert info["shield_reason"] == "restricted"
    assert info["executed_action"] == 2
    assert info["cost_restricted"] == 0.0
    assert info["cost"] == 0.0


def test_projection_is_independent_of_hazard_location() -> None:
    executed_actions: list[int] = []

    for scenario_id, hazards in (
        ("hazard_north", ((0, 0),)),
        ("hazard_south", ((5, 5),)),
    ):
        environment = SafetyShield(
            ConstructionInspectionEnv(
                scenario=create_scenario(
                    scenario_id=scenario_id,
                    hazards=hazards,
                    workers=((2, 4),),
                )
            )
        )
        environment.reset(seed=0)

        _, _, _, _, info = environment.step(3)
        executed_actions.append(int(info["executed_action"]))

    assert executed_actions == [0, 0]


def test_shield_exposes_task_action_masks() -> None:
    base_environment = ConstructionInspectionEnv(
        scenario=create_scenario(
            scenario_id="shield_masks",
        )
    )
    environment = SafetyShield(base_environment)
    environment.reset(seed=0)

    assert np.array_equal(
        environment.action_masks(),
        base_environment.action_masks(),
    )


def test_maskable_ppo_trains_through_semantic_shield() -> None:
    scenario = create_scenario(
        scenario_id="shield_maskable_integration",
        workers=((2, 4),),
        restricted_zones=((3, 2),),
    )
    environment = DummyVecEnv(
        [
            lambda: Monitor(
                SafetyShield(
                    ConstructionInspectionEnv(
                        scenario=scenario,
                    )
                )
            )
        ]
    )

    try:
        model = MaskablePPO(
            "MultiInputPolicy",
            environment,
            n_steps=8,
            batch_size=8,
            n_epochs=1,
            learning_rate=1e-3,
            seed=0,
            verbose=0,
        )
        model.learn(total_timesteps=8)

        observation = environment.reset()
        masks = get_action_masks(environment)
        action, _ = model.predict(
            observation,
            action_masks=masks,
            deterministic=True,
        )

        selected_action = int(np.asarray(action).item())

        assert bool(masks[0, selected_action])
    finally:
        environment.close()
