from __future__ import annotations

import argparse
import json
import random
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from riskaware_saferrl.callbacks import (
    ActionDiagnosticsCallback,
    ScenarioEvaluationCallback,
)
from riskaware_saferrl.envs import ConstructionInspectionEnv
from riskaware_saferrl.evaluation.scenario_evaluator import (
    evaluate_policy_on_scenarios,
    save_evaluation_results,
)
from riskaware_saferrl.models import SemanticMapExtractor
from riskaware_saferrl.safety import SafetyShield
from riskaware_saferrl.scenario_dataset import load_scenarios


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", type=Path, default=Path("data/scenarios"))
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--validation-split", type=str, default="validation")
    parser.add_argument("--updates", type=int, default=100)
    parser.add_argument("--n-steps", type=int, default=256)
    parser.add_argument("--n-epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--timesteps", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-envs", type=int, default=4)
    parser.add_argument("--device", choices=["cpu", "cuda", "auto"], default="auto")
    parser.add_argument("--shield", action="store_true")
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--resume-from", type=Path, default=None)
    parser.add_argument("--eval-every-updates", type=int, default=10)
    parser.add_argument("--checkpoint-every-updates", type=int, default=10)
    parser.add_argument("--validation-limit", type=int, default=50)
    parser.add_argument("--final-validation-limit", type=int, default=None)
    parser.add_argument(
        "--selection-metric",
        choices=["reward", "hazard_recall", "safe_hazard_recall"],
        default="safe_hazard_recall",
    )
    parser.add_argument("--safety-cost-limit", type=float, default=5.0)
    parser.add_argument("--progress-bar", action="store_true")
    parser.add_argument("--smoke-test", action="store_true")
    return parser.parse_args()


def build_env(scenarios, use_shield: bool):
    def create_environment():
        environment = ConstructionInspectionEnv(scenarios=scenarios)
        if use_shield:
            environment = SafetyShield(environment)
        return Monitor(environment)

    return create_environment


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def validate_args(args: argparse.Namespace) -> None:
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but PyTorch cannot access the NVIDIA GPU.")
    if args.n_envs < 1:
        raise ValueError("--n-envs must be positive.")
    if args.n_steps < 8:
        raise ValueError("--n-steps must be at least 8.")
    if args.n_epochs < 1:
        raise ValueError("--n-epochs must be positive.")
    if args.batch_size < 2:
        raise ValueError("--batch-size must be at least 2.")

    rollout_size = args.n_steps * args.n_envs
    if args.batch_size > rollout_size:
        raise ValueError("--batch-size cannot exceed the rollout size.")
    if rollout_size % args.batch_size != 0:
        raise ValueError("The rollout size must be divisible by --batch-size.")
    if args.eval_every_updates < 1:
        raise ValueError("--eval-every-updates must be positive.")
    if args.checkpoint_every_updates < 1:
        raise ValueError("--checkpoint-every-updates must be positive.")
    if args.validation_limit < 1:
        raise ValueError("--validation-limit must be positive.")
    if not args.smoke_test and args.updates < 100:
        raise ValueError(
            "Full training requires at least 100 rollout updates. "
            "Use --smoke-test only for implementation validation."
        )


def create_model(
    args: argparse.Namespace,
    environment: DummyVecEnv,
    tensorboard_directory: Path,
) -> PPO:
    if args.resume_from is not None:
        model = PPO.load(
            args.resume_from,
            env=environment,
            device=args.device,
            tensorboard_log=str(tensorboard_directory),
        )
        if model.n_steps != args.n_steps:
            raise ValueError("The resumed model n_steps does not match --n-steps.")
        if model.n_epochs != args.n_epochs:
            raise ValueError("The resumed model n_epochs does not match --n-epochs.")
        if model.batch_size != args.batch_size:
            raise ValueError("The resumed model batch_size does not match --batch-size.")
        return model

    policy_kwargs = {
        "features_extractor_class": SemanticMapExtractor,
        "features_extractor_kwargs": {"features_dim": 256},
        "net_arch": {"pi": [128, 64], "vf": [128, 64]},
    }

    return PPO(
        policy="MultiInputPolicy",
        env=environment,
        learning_rate=3e-4,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
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


def main() -> None:
    args = parse_args()
    validate_args(args)
    set_global_seed(args.seed)

    train_scenarios = load_scenarios(args.dataset_dir, args.split, verify_hash=True)
    validation_scenarios = load_scenarios(
        args.dataset_dir,
        args.validation_split,
        verify_hash=True,
    )

    if args.smoke_test:
        effective_updates = 2
        validation_limit = min(5, len(validation_scenarios))
        eval_every_updates = 1
        checkpoint_every_updates = 1
        final_validation_limit = validation_limit
    else:
        effective_updates = args.updates
        validation_limit = min(args.validation_limit, len(validation_scenarios))
        eval_every_updates = args.eval_every_updates
        checkpoint_every_updates = args.checkpoint_every_updates
        final_validation_limit = (
            len(validation_scenarios)
            if args.final_validation_limit is None
            else min(args.final_validation_limit, len(validation_scenarios))
        )

    rollout_size = args.n_steps * args.n_envs
    total_timesteps = (
        args.timesteps if args.timesteps is not None else rollout_size * effective_updates
    )
    if not args.smoke_test and total_timesteps < rollout_size * 100:
        raise ValueError("Full training must include at least 100 rollout updates.")

    if args.run_name is not None:
        run_name = args.run_name
    elif args.shield:
        run_name = f"ppo_{args.split}_shielded_seed{args.seed}"
    else:
        run_name = f"ppo_{args.split}_seed{args.seed}"

    run_directory = Path("artifacts/runs") / run_name
    checkpoint_directory = run_directory / "checkpoints"
    evaluation_directory = run_directory / "evaluations"
    diagnostics_directory = run_directory / "diagnostics"
    tensorboard_directory = Path("artifacts/tensorboard")

    for directory in (
        run_directory,
        checkpoint_directory,
        evaluation_directory,
        diagnostics_directory,
        tensorboard_directory,
    ):
        directory.mkdir(parents=True, exist_ok=True)

    environment = DummyVecEnv([build_env(train_scenarios, args.shield) for _ in range(args.n_envs)])
    environment.seed(args.seed)
    model = create_model(args, environment, tensorboard_directory)

    metadata: dict[str, Any] = {
        "run_name": run_name,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "git_commit": get_git_commit(),
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_runtime": torch.version.cuda,
        "gpu_name": (torch.cuda.get_device_name(0) if torch.cuda.is_available() else None),
        "seed": args.seed,
        "train_split": args.split,
        "validation_split": args.validation_split,
        "train_scenario_count": len(train_scenarios),
        "validation_scenario_count": validation_limit,
        "n_envs": args.n_envs,
        "n_steps": args.n_steps,
        "rollout_size": rollout_size,
        "rollout_updates": effective_updates,
        "n_epochs": args.n_epochs,
        "total_optimization_epochs": effective_updates * args.n_epochs,
        "batch_size": args.batch_size,
        "total_timesteps": total_timesteps,
        "shield": args.shield,
        "selection_metric": args.selection_metric,
        "safety_cost_limit": args.safety_cost_limit,
        "resume_from": str(args.resume_from) if args.resume_from else None,
        "smoke_test": args.smoke_test,
    }
    (run_directory / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n",
        encoding="utf-8",
    )

    callbacks = CallbackList(
        [
            CheckpointCallback(
                save_freq=max(checkpoint_every_updates * args.n_steps, 1),
                save_path=str(checkpoint_directory),
                name_prefix=run_name,
                verbose=2,
            ),
            ActionDiagnosticsCallback(diagnostics_directory / "rollout_diagnostics.jsonl"),
            ScenarioEvaluationCallback(
                validation_scenarios[:validation_limit],
                eval_freq=max(eval_every_updates * args.n_steps, 1),
                output_directory=evaluation_directory,
                selection_metric=args.selection_metric,
                safety_cost_limit=args.safety_cost_limit,
                use_shield=args.shield,
                evaluate_at_start=True,
                verbose=1,
            ),
        ]
    )

    print(json.dumps(metadata, indent=2))
    model.learn(
        total_timesteps=total_timesteps,
        callback=callbacks,
        tb_log_name=run_name,
        reset_num_timesteps=args.resume_from is None,
        progress_bar=args.progress_bar,
    )

    final_model_path = run_directory / "final_model"
    model.save(final_model_path)

    records, summary = evaluate_policy_on_scenarios(
        model,
        validation_scenarios[:final_validation_limit],
        use_shield=args.shield,
        deterministic=True,
    )
    summary["timesteps"] = int(model.num_timesteps)
    summary["evaluation_type"] = "final_validation"
    save_evaluation_results(
        records,
        summary,
        evaluation_directory,
        "final_validation",
    )
    environment.close()

    print(f"Final model: {final_model_path}.zip")
    print(f"Best model: {evaluation_directory / 'best_model' / 'best_model'}.zip")
    print(f"Diagnostics: {diagnostics_directory / 'rollout_diagnostics.jsonl'}")


if __name__ == "__main__":
    main()
