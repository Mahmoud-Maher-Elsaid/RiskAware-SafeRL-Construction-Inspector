import argparse
from pathlib import Path

import gymnasium as gym
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

import riskaware_saferrl
from riskaware_saferrl.models import SemanticMapExtractor
from riskaware_saferrl.safety import SafetyShield


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--timesteps",
        type=int,
        default=100_000,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    parser.add_argument(
        "--n-envs",
        type=int,
        default=4,
    )

    parser.add_argument(
        "--device",
        choices=["cpu", "cuda", "auto"],
        default="auto",
    )

    parser.add_argument(
        "--shield",
        action="store_true",
    )

    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
    )

    return parser.parse_args()


def build_env(use_shield: bool):
    def create_environment():
        env = gym.make(riskaware_saferrl.ENV_ID)

        if use_shield:
            env = SafetyShield(env)

        return Monitor(env)

    return create_environment


def main() -> None:
    args = parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but PyTorch cannot access the NVIDIA GPU.")

    model_directory = Path("artifacts/models")
    tensorboard_directory = Path("artifacts/tensorboard")

    model_directory.mkdir(parents=True, exist_ok=True)
    tensorboard_directory.mkdir(parents=True, exist_ok=True)

    environment = DummyVecEnv([build_env(args.shield) for _ in range(args.n_envs)])

    environment.seed(args.seed)

    policy_kwargs = {
        "features_extractor_class": SemanticMapExtractor,
        "features_extractor_kwargs": {
            "features_dim": 256,
        },
        "net_arch": {
            "pi": [128, 64],
            "vf": [128, 64],
        },
    }

    model = PPO(
        policy="MultiInputPolicy",
        env=environment,
        learning_rate=3e-4,
        n_steps=256,
        batch_size=256,
        n_epochs=5,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        policy_kwargs=policy_kwargs,
        verbose=1,
        tensorboard_log=str(tensorboard_directory),
        seed=args.seed,
        device=args.device,
    )

    if args.run_name is not None:
        run_name = args.run_name
    elif args.shield:
        run_name = "ppo_shielded"
    else:
        run_name = "ppo_baseline"

    model.learn(
        total_timesteps=args.timesteps,
        tb_log_name=run_name,
    )

    model_path = model_directory / run_name
    model.save(model_path)

    environment.close()

    print(f"Saved model to {model_path}.zip")


if __name__ == "__main__":
    main()
