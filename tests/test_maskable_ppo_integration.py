import numpy as np
from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.utils import get_action_masks
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from riskaware_saferrl.envs import ConstructionInspectionEnv
from riskaware_saferrl.scenarios import Scenario


def test_maskable_ppo_trains_with_environment_action_masks() -> None:
    scenario = Scenario(
        scenario_id="maskable_integration",
        split="test",
        grid_size=6,
        agent_start=(0, 0),
        obstacles=((1, 0),),
        hazards=((0, 2),),
        workers=(),
        restricted_zones=(),
        max_steps=8,
        vision_radius=3,
    )

    environment = DummyVecEnv(
        [
            lambda: Monitor(
                ConstructionInspectionEnv(
                    scenario=scenario,
                    inspection_radius=1,
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
