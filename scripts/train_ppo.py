import argparse
from pathlib import Path

import torch
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from riskaware_saferrl.envs import ConstructionInspectionEnv
from riskaware_saferrl.models import SemanticMapExtractor
from riskaware_saferrl.safety import SafetyShield
from riskaware_saferrl.scenario_dataset import load_scenarios


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=Path("data/scenarios"),
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
    )
    parser.add_argument(
        "--updates",
        type=int,
        default=100,
    )
    parser.add_argument(
        "--n-steps",
        type=int,
        default=256,
    )
    parser.add_argument(
        "--n-epochs",
        type=int,
        default=10,
    )
    parser.add_argument(
        "--timesteps",
        type=int,
        default=None,
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
    parser.add_argument(
        "--smoke-test",
        action="store_true",
    )

    return parser.parse_args()


def build_env(
    scenarios,
    use_shield: bool,
):
    def create_environment():
        environment = ConstructionInspectionEnv(scenarios=scenarios)

        if use_shield:
            environment = SafetyShield(environment)

        return Monitor(environment)

    return create_environment


def main() -> None:
    args = parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but PyTorch cannot access the NVIDIA GPU.")

    if args.n_envs < 1:
        raise ValueError("--n-envs must be positive.")

    if args.n_steps < 8:
        raise ValueError("--n-steps must be at least 8.")

    if args.n_epochs < 1:
        raise ValueError("--n-epochs must be positive.")

    if not args.smoke_test and args.updates < 100:
        raise ValueError(
            "Full training requires at least 100 rollout updates. "
            "Use --smoke-test only for implementation validation."
        )

    scenarios = load_scenarios(
        args.dataset_dir,
        args.split,
        verify_hash=True,
    )

    model_directory = Path("artifacts/models")
    tensorboard_directory = Path("artifacts/tensorboard")

    model_directory.mkdir(parents=True, exist_ok=True)
    tensorboard_directory.mkdir(parents=True, exist_ok=True)

    environment = DummyVecEnv(
        [
            build_env(
                scenarios,
                args.shield,
            )
            for _ in range(args.n_envs)
        ]
    )
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

    effective_updates = 2 if args.smoke_test else args.updates
    rollout_size = args.n_steps * args.n_envs
    total_timesteps = (
        args.timesteps if args.timesteps is not None else rollout_size * effective_updates
    )

    if not args.smoke_test and total_timesteps < rollout_size * 100:
        raise ValueError("Full training must include at least 100 rollout updates.")

    model = PPO(
        policy="MultiInputPolicy",
        env=environment,
        learning_rate=3e-4,
        n_steps=args.n_steps,
        batch_size=256,
        n_epochs=args.n_epochs,
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
        run_name = f"ppo_{args.split}_shielded_seed{args.seed}"
    else:
        run_name = f"ppo_{args.split}_seed{args.seed}"

    print(
        {
            "split": args.split,
            "scenario_count": len(scenarios),
            "n_envs": args.n_envs,
            "n_steps": args.n_steps,
            "rollout_size": rollout_size,
            "rollout_updates": effective_updates,
            "optimization_epochs_per_update": args.n_epochs,
            "total_optimization_epochs": effective_updates * args.n_epochs,
            "total_timesteps": total_timesteps,
            "smoke_test": args.smoke_test,
        }
    )

    model.learn(
        total_timesteps=total_timesteps,
        tb_log_name=run_name,
    )

    model_path = model_directory / run_name
    model.save(model_path)
    environment.close()

    print(f"Saved model to {model_path}.zip")


if __name__ == "__main__":
    main()
