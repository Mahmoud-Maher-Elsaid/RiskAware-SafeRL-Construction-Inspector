from __future__ import annotations

import argparse
import hashlib
import json
import random
import subprocess
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import torch
import yaml
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback
from stable_baselines3.common.vec_env import DummyVecEnv

from riskaware_saferrl.training import (
    EpisodeMetricsCallback,
    flatten_research_environment,
    load_grid_config,
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/training/ppo.yaml"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/final_submission/ppo"))
    parser.add_argument("--timesteps", type=int)
    args = parser.parse_args()
    settings = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    seed = int(settings["seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    config = load_grid_config(Path(settings["environment"]))
    environment = DummyVecEnv(
        [lambda: flatten_research_environment(config, continuous_actions=False)]
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    final_checkpoint = args.output_dir / "best_model.zip"
    resume_checkpoint = args.output_dir / "resume_model.zip"
    if resume_checkpoint.exists() and settings.get("resume", True):
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
    timesteps = int(args.timesteps or settings["total_timesteps"])
    callbacks = CallbackList(
        [
            CheckpointCallback(
                save_freq=int(settings["checkpoint_frequency"]),
                save_path=str(args.output_dir / "checkpoints"),
                name_prefix="ppo",
            ),
            EpisodeMetricsCallback(
                args.output_dir / "training_metrics.jsonl",
                early_stopping_patience_episodes=int(settings["early_stopping_patience_episodes"]),
                minimum_episodes=int(settings["minimum_episodes_before_stopping"]),
            ),
        ]
    )
    model.learn(
        total_timesteps=timesteps,
        callback=callbacks,
        reset_num_timesteps=reset_num_timesteps,
        tb_log_name="ppo",
    )
    model.save(final_checkpoint)
    model.save(resume_checkpoint)
    metadata = {
        "algorithm": "PPO",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "seed": seed,
        "timesteps": int(model.num_timesteps),
        "device": str(model.device),
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "observation_space": str(environment.observation_space),
        "action_space": str(environment.action_space),
        "checkpoint": str(final_checkpoint),
        "checkpoint_sha256": sha256(final_checkpoint),
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
