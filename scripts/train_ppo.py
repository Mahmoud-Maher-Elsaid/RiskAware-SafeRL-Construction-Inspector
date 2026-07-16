import argparse
from pathlib import Path

import gymnasium as gym
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

import riskaware_saferrl
from riskaware_saferrl.safety import SafetyShield


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--timesteps", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-envs", type=int, default=4)
    parser.add_argument("--device", choices=["cpu", "cuda", "auto"], default="auto")
    parser.add_argument("--shield", action="store_true")
    return parser.parse_args()


def build_env(use_shield: bool):
    def _factory():
        env = gym.make(riskaware_saferrl.ENV_ID)
        if use_shield:
            env = SafetyShield(env)
        return Monitor(env)

    return _factory


def main() -> None:
    args = parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but PyTorch cannot access the NVIDIA GPU.")

    output_dir = Path("artifacts/models")
    tensorboard_dir = Path("artifacts/tensorboard")
    output_dir.mkdir(parents=True, exist_ok=True)
    tensorboard_dir.mkdir(parents=True, exist_ok=True)

    env = DummyVecEnv([build_env(args.shield) for _ in range(args.n_envs)])
    env.seed(args.seed)

    model = PPO(
        policy="MultiInputPolicy",
        env=env,
        learning_rate=3e-4,
        n_steps=256,
        batch_size=256,
        n_epochs=5,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        verbose=1,
        tensorboard_log=str(tensorboard_dir),
        seed=args.seed,
        device=args.device,
    )

    run_name = "ppo_shielded" if args.shield else "ppo_baseline"
    model.learn(total_timesteps=args.timesteps, tb_log_name=run_name)
    model.save(output_dir / run_name)
    env.close()

    print(f"Saved model to {output_dir / run_name}.zip")


if __name__ == "__main__":
    main()