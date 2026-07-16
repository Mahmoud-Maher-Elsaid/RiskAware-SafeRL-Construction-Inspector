import numpy as np
from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.utils import (
    get_action_masks,
)
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from riskaware_saferrl.envs import (
    ConstructionInspectionEnv,
)
from riskaware_saferrl.safety import SafetyShield
from riskaware_saferrl.scenarios import Scenario


def create_scenario(
    *,
    scenario_id: str,
    hazards: tuple[
        tuple[int, int],
        ...,
    ] = ((0, 0),),
    workers: tuple[
        tuple[int, int],
        ...,
    ] = (),
    restricted_zones: tuple[
        tuple[int, int],
        ...,
    ] = (),
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


def test_inspect_reports_current_worker_risk() -> None:
    environment = ConstructionInspectionEnv(
        scenario=create_scenario(
            scenario_id="inspect_worker_risk",
            workers=((2, 3),),
        )
    )
    environment.reset(seed=0)

    assert environment.action_safety_violations(4) == ("worker",)
    assert not environment.is_action_safe(4)


def test_environment_reports_combined_violation() -> None:
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


def test_shield_preserves_safe_action() -> None:
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
    assert info["shield_resolution"] == "not_needed"
    assert info["executed_action"] == 0


def test_shield_projects_worker_risk() -> None:
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
    assert info["shield_resolution"] == ("safe_projection")
    assert info["executed_action"] == 0
    assert info["shield_replacement_safe"] is True
    assert info["cost"] == 0.0


def test_shield_uses_safe_emergency_hold() -> None:
    environment = SafetyShield(
        ConstructionInspectionEnv(
            scenario=create_scenario(
                scenario_id="emergency_hold",
                restricted_zones=(
                    (1, 2),
                    (3, 2),
                    (2, 1),
                    (2, 3),
                ),
            )
        )
    )
    environment.reset(seed=0)

    _, _, _, _, info = environment.step(0)

    assert info["shield_active"] is True
    assert info["shield_resolution"] == ("emergency_hold")
    assert info["executed_action"] == 4
    assert info["shield_emergency_hold"] is True
    assert info["shield_replacement_task_valid"] is False
    assert info["shield_replacement_safe"] is True
    assert info["cost"] == 0.0


def test_shield_handles_unavoidable_violation() -> None:
    environment = SafetyShield(
        ConstructionInspectionEnv(
            scenario=create_scenario(
                scenario_id="least_unsafe",
                workers=(
                    (1, 2),
                    (3, 2),
                    (2, 1),
                    (2, 3),
                ),
            )
        )
    )
    environment.reset(seed=0)

    _, _, _, _, info = environment.step(0)

    assert info["shield_active"] is True
    assert info["shield_resolution"] == ("least_unsafe")
    assert info["shield_unavoidable_violation"] is True
    assert info["executed_action"] in (0, 1, 2, 3)


def test_projection_is_hazard_independent() -> None:
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


def test_shield_exposes_task_masks() -> None:
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


def test_maskable_ppo_trains_through_shield() -> None:
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
