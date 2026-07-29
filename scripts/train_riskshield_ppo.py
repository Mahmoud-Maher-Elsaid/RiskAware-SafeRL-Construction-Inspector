from __future__ import annotations

import argparse
import hashlib
import json
import random
import subprocess
from datetime import UTC, datetime
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
import yaml
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback
from stable_baselines3.common.vec_env import DummyVecEnv

from riskaware_saferrl.algorithms import RiskShieldCostWrapper
from riskaware_saferrl.envs import ResearchConstructionEnv
from riskaware_saferrl.safety import PredictiveSafetyShield, PredictiveShieldWrapper
from riskaware_saferrl.training import EpisodeMetricsCallback, load_grid_config


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/training/riskshield_ppo.yaml"))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/final_submission/riskshield_ppo"),
    )
    parser.add_argument("--timesteps", type=int)
    args = parser.parse_args()
    settings = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    seed = int(settings["seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    grid_config = load_grid_config(Path(settings["environment"]))

    def create_environment():
        environment = ResearchConstructionEnv(grid_config)
        if settings["shield_during_training"]:
            environment = PredictiveShieldWrapper(
                environment,
                PredictiveSafetyShield(
                    horizon=int(settings["predictive_shield_horizon"]),
                    safety_budget=float(settings["safety_budget"]),
                ),
            )
        environment = RiskShieldCostWrapper(
            environment,
            safety_budget=float(settings["safety_budget"]),
            lagrange_learning_rate=float(settings["lagrange_learning_rate"]),
            cost_learning_rate=float(settings["cost_value_learning_rate"]),
            gamma=float(settings["gamma"]),
            device=settings["device"],
        )
        return gym.wrappers.FlattenObservation(environment)

    environment = DummyVecEnv([create_environment])
    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = args.output_dir / "best_model.zip"
    resume_checkpoint = args.output_dir / "resume_model.zip"
    if resume_checkpoint.exists() and settings["resume"]:
        model = PPO.load(resume_checkpoint, env=environment, device=settings["device"])
        reset_num_timesteps = False
    else:
        model = PPO(
            "MlpPolicy",
            environment,
            learning_rate=float(settings["learning_rate"]),
            n_steps=int(settings["n_steps"]),
            batch_size=int(settings["batch_size"]),
            n_epochs=int(settings["n_epochs"]),
            gamma=float(settings["gamma"]),
            gae_lambda=float(settings["gae_lambda"]),
            seed=seed,
            device=settings["device"],
            tensorboard_log=str(args.output_dir / "tensorboard"),
            verbose=1,
        )
        reset_num_timesteps = True
    callbacks = CallbackList(
        [
            CheckpointCallback(
                save_freq=int(settings["checkpoint_frequency"]),
                save_path=str(args.output_dir / "checkpoints"),
                name_prefix="riskshield_ppo",
            ),
            EpisodeMetricsCallback(
                args.output_dir / "training_metrics.jsonl",
                early_stopping_patience_episodes=int(settings["early_stopping_patience_episodes"]),
                minimum_episodes=int(settings["minimum_episodes_before_stopping"]),
                verbose=1,
            ),
        ]
    )
    timesteps = int(args.timesteps or settings["total_timesteps"])
    model.learn(
        total_timesteps=timesteps,
        callback=callbacks,
        reset_num_timesteps=reset_num_timesteps,
        tb_log_name="riskshield_ppo",
    )
    model.save(checkpoint)
    model.save(resume_checkpoint)
    cost_wrapper: RiskShieldCostWrapper = environment.envs[0].env
    cost_checkpoint = args.output_dir / "cost_value.pt"
    cost_wrapper.save_cost_value(str(cost_checkpoint))
    metadata = {
        "algorithm": "RiskShield-PPO",
        "base_algorithm": "PPO",
        "constrained_method": "PPO-Lagrangian with learned cost value",
        "objective": "maximize E[return] - lambda * E[safety_cost]",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "seed": seed,
        "timesteps": int(model.num_timesteps),
        "device": str(model.device),
        "safety_budget": float(settings["safety_budget"]),
        "lagrangian_multiplier": float(cost_wrapper.lagrange.value),
        "cost_value_loss": float(cost_wrapper.last_cost_value_loss),
        "shield_during_training": bool(settings["shield_during_training"]),
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256(checkpoint),
        "cost_value_checkpoint": str(cost_checkpoint),
        "cost_value_sha256": sha256(cost_checkpoint),
        "config": str(args.config),
        "smoke_test": timesteps < int(settings["total_timesteps"]),
    }
    (args.output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(metadata, indent=2))
    environment.close()


if __name__ == "__main__":
    main()
