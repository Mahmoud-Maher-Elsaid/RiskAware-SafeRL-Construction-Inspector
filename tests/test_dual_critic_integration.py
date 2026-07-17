from __future__ import annotations

import numpy as np
import torch
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import (
    DummyVecEnv,
)

from riskaware_saferrl.algorithms import (
    DualCriticMaskablePPO,
)
from riskaware_saferrl.envs import (
    ConstructionInspectionEnv,
)
from riskaware_saferrl.models import (
    SemanticMapExtractor,
)
from riskaware_saferrl.scenarios import Scenario


def make_environment():
    scenario = Scenario(
        scenario_id="dual_critic_test",
        split="test",
        grid_size=6,
        agent_start=(2, 2),
        obstacles=((0, 5),),
        hazards=((2, 3),),
        workers=((3, 2),),
        restricted_zones=((1, 2),),
        max_steps=8,
        vision_radius=3,
    )
    return Monitor(
        ConstructionInspectionEnv(
            scenario=scenario,
            inspection_radius=2,
        )
    )


def test_dual_critic_learns_one_short_rollout() -> None:
    torch.manual_seed(0)
    np.random.seed(0)

    environment = DummyVecEnv([make_environment])
    model = DualCriticMaskablePPO(
        policy="MultiInputPolicy",
        env=environment,
        learning_rate=3e-4,
        n_steps=8,
        batch_size=8,
        n_epochs=1,
        gamma=0.99,
        gae_lambda=0.95,
        policy_kwargs={
            "features_extractor_class": (SemanticMapExtractor),
            "features_extractor_kwargs": {
                "features_dim": 32,
            },
            "net_arch": {
                "pi": [16],
                "vf": [16],
            },
        },
        cost_limit=1.0,
        cost_gamma=1.0,
        cost_gae_lambda=0.95,
        cost_learning_rate=1e-3,
        cost_features_dim=32,
        dual_learning_rate=0.01,
        dual_maximum=1.0,
        dual_ema_beta=0.0,
        dual_warmup_updates=0,
        seed=0,
        device="cpu",
        verbose=0,
    )

    before = [parameter.detach().clone() for parameter in model.cost_value_network.parameters()]

    model.learn(total_timesteps=16)

    after = list(model.cost_value_network.parameters())
    assert any(
        not torch.equal(
            first,
            second.detach(),
        )
        for first, second in zip(
            before,
            after,
            strict=True,
        )
    )
    assert model.get_dual_diagnostics()["completed_episodes"] >= 1

    environment.close()
