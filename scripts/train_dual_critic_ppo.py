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
from sb3_contrib import MaskablePPO
from stable_baselines3.common.callbacks import (
    CallbackList,
    CheckpointCallback,
)
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import (
    DummyVecEnv,
)

from riskaware_saferrl.algorithms import (
    DualCriticMaskablePPO,
)
from riskaware_saferrl.callbacks.action_diagnostics import (
    ActionDiagnosticsCallback,
)
from riskaware_saferrl.callbacks.curriculum_progress import (
    CurriculumProgressCallback,
)
from riskaware_saferrl.callbacks.dual_critic_diagnostics import (
    DualCriticDiagnosticsCallback,
)
from riskaware_saferrl.callbacks.scenario_evaluation import (
    ScenarioEvaluationCallback,
)
from riskaware_saferrl.curriculum import (
    feasible_validation_scenarios_from_manifest,
    load_curriculum_manifest,
    scenario_tiers_from_manifest,
    sha256_file,
)
from riskaware_saferrl.envs.curriculum_env import (
    CurriculumConstructionInspectionEnv,
)
from riskaware_saferrl.evaluation.scenario_evaluator import (
    evaluate_policy_on_scenarios,
    save_evaluation_results,
)
from riskaware_saferrl.models import (
    SemanticMapExtractor,
)
from riskaware_saferrl.scenario_dataset import (
    load_scenarios,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=Path("data/scenarios"),
    )
    parser.add_argument(
        "--curriculum-manifest",
        type=Path,
        default=Path("configs/curriculum/safe_viewpoint_radius2.json"),
    )
    parser.add_argument(
        "--updates",
        type=int,
        default=100,
    )
    parser.add_argument(
        "--easy-updates",
        type=int,
        default=25,
    )
    parser.add_argument(
        "--medium-updates",
        type=int,
        default=25,
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
        "--batch-size",
        type=int,
        default=256,
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
        "--run-name",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--policy-init-from",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--eval-every-updates",
        type=int,
        default=10,
    )
    parser.add_argument(
        "--checkpoint-every-updates",
        type=int,
        default=10,
    )
    parser.add_argument(
        "--validation-limit",
        type=int,
        default=50,
    )
    parser.add_argument(
        "--final-validation-limit",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--full-validation-limit",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--selection-metric",
        choices=[
            "reward",
            "hazard_recall",
            "safe_hazard_recall",
        ],
        default="safe_hazard_recall",
    )
    parser.add_argument(
        "--cost-limit",
        type=float,
        default=5.0,
    )
    parser.add_argument(
        "--cost-gamma",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--cost-gae-lambda",
        type=float,
        default=0.95,
    )
    parser.add_argument(
        "--cost-learning-rate",
        type=float,
        default=1e-4,
    )
    parser.add_argument(
        "--cost-features-dim",
        type=int,
        default=128,
    )
    parser.add_argument(
        "--initial-lagrange-multiplier",
        type=float,
        default=0.0,
    )
    parser.add_argument(
        "--dual-learning-rate",
        type=float,
        default=0.01,
    )
    parser.add_argument(
        "--dual-maximum",
        type=float,
        default=2.0,
    )
    parser.add_argument(
        "--dual-ema-beta",
        type=float,
        default=0.9,
    )
    parser.add_argument(
        "--dual-warmup-updates",
        type=int,
        default=5,
    )
    parser.add_argument(
        "--target-kl",
        type=float,
        default=0.03,
    )
    parser.add_argument(
        "--progress-bar",
        action="store_true",
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
    )
    parser.add_argument(
        "--calibration-run",
        action="store_true",
    )

    return parser.parse_args()


def validate_args(
    args: argparse.Namespace,
) -> None:
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but PyTorch cannot access the NVIDIA GPU.")
    if args.smoke_test and args.calibration_run:
        raise ValueError("--smoke-test and --calibration-run cannot be used together.")
    if args.n_envs < 1:
        raise ValueError("--n-envs must be positive.")
    if args.n_steps < 8:
        raise ValueError("--n-steps must be at least 8.")
    if args.n_epochs < 1:
        raise ValueError("--n-epochs must be positive.")
    if args.batch_size < 2:
        raise ValueError("--batch-size must be at least 2.")
    if args.easy_updates < 1:
        raise ValueError("--easy-updates must be positive.")
    if args.medium_updates < 1:
        raise ValueError("--medium-updates must be positive.")
    if args.eval_every_updates < 1:
        raise ValueError("--eval-every-updates must be positive.")
    if args.checkpoint_every_updates < 1:
        raise ValueError("--checkpoint-every-updates must be positive.")
    if args.validation_limit < 1:
        raise ValueError("--validation-limit must be positive.")
    if args.cost_limit < 0.0:
        raise ValueError("--cost-limit must be non-negative.")
    if not 0.0 <= args.cost_gamma <= 1.0:
        raise ValueError("--cost-gamma must be in [0, 1].")
    if not 0.0 <= args.cost_gae_lambda <= 1.0:
        raise ValueError("--cost-gae-lambda must be in [0, 1].")
    if args.cost_learning_rate <= 0.0:
        raise ValueError("--cost-learning-rate must be positive.")
    if args.dual_learning_rate <= 0.0:
        raise ValueError("--dual-learning-rate must be positive.")
    if args.dual_maximum <= 0.0:
        raise ValueError("--dual-maximum must be positive.")
    if not 0.0 <= args.dual_ema_beta < 1.0:
        raise ValueError("--dual-ema-beta must be in [0, 1).")
    if args.dual_warmup_updates < 0:
        raise ValueError("--dual-warmup-updates must be non-negative.")
    if args.target_kl <= 0.0:
        raise ValueError("--target-kl must be positive.")
    if args.policy_init_from is not None and not args.policy_init_from.exists():
        raise FileNotFoundError(
            f"Policy initialization checkpoint was not found: {args.policy_init_from}"
        )

    rollout_size = args.n_steps * args.n_envs
    if args.batch_size > rollout_size:
        raise ValueError("--batch-size cannot exceed the rollout size.")
    if rollout_size % args.batch_size != 0:
        raise ValueError("The rollout size must be divisible by --batch-size.")

    if not args.smoke_test:
        total_updates = args.updates
        if not args.calibration_run and total_updates < 100:
            raise ValueError(
                "Full training requires at least "
                "100 rollout updates. Use "
                "--calibration-run for a shorter "
                "diagnostic experiment."
            )
        if args.easy_updates + args.medium_updates >= total_updates:
            raise ValueError("The curriculum schedule must leave at least one full-stage update.")


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
    except (
        OSError,
        subprocess.CalledProcessError,
    ):
        return None


def build_env(
    scenario_tiers,
    *,
    inspection_radius: int,
):
    def create_environment():
        environment = CurriculumConstructionInspectionEnv(
            scenario_tiers,
            inspection_radius=inspection_radius,
        )
        return Monitor(environment)

    return create_environment


def create_model(
    args: argparse.Namespace,
    environment: DummyVecEnv,
    tensorboard_directory: Path,
) -> DualCriticMaskablePPO:
    policy_kwargs = {
        "features_extractor_class": (SemanticMapExtractor),
        "features_extractor_kwargs": {
            "features_dim": 256,
        },
        "net_arch": {
            "pi": [128, 64],
            "vf": [128, 64],
        },
    }

    model = DualCriticMaskablePPO(
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
        vf_coef=0.5,
        max_grad_norm=0.5,
        target_kl=args.target_kl,
        tensorboard_log=str(tensorboard_directory),
        policy_kwargs=policy_kwargs,
        verbose=1,
        seed=args.seed,
        device=args.device,
        cost_limit=args.cost_limit,
        cost_gamma=args.cost_gamma,
        cost_gae_lambda=args.cost_gae_lambda,
        cost_learning_rate=(args.cost_learning_rate),
        cost_features_dim=(args.cost_features_dim),
        initial_lagrange_multiplier=(args.initial_lagrange_multiplier),
        dual_learning_rate=(args.dual_learning_rate),
        dual_maximum=args.dual_maximum,
        dual_ema_beta=args.dual_ema_beta,
        dual_warmup_updates=(args.dual_warmup_updates),
    )

    if args.policy_init_from is not None:
        baseline_model = MaskablePPO.load(
            args.policy_init_from,
            device=args.device,
        )
        model.policy.load_state_dict(
            baseline_model.policy.state_dict(),
            strict=True,
        )
        del baseline_model
        print(f"Policy initialized from: {args.policy_init_from}")

    return model


def save_final_evaluation(
    *,
    model: DualCriticMaskablePPO,
    scenarios,
    use_shield: bool,
    output_directory: Path,
    output_name: str,
) -> dict[str, Any]:
    records, summary = evaluate_policy_on_scenarios(
        model,
        scenarios,
        use_shield=use_shield,
        use_action_masks=True,
        deterministic=True,
    )
    summary["timesteps"] = int(model.num_timesteps)
    summary["evaluation_type"] = output_name
    save_evaluation_results(
        records,
        summary,
        output_directory,
        output_name,
    )
    return summary


def main() -> None:
    args = parse_args()
    validate_args(args)
    set_global_seed(args.seed)

    source_manifest_path = args.dataset_dir / "manifest.json"
    curriculum_manifest = load_curriculum_manifest(
        args.curriculum_manifest,
        source_manifest_path=(source_manifest_path),
        verify_source_hash=True,
    )
    train_split = str(curriculum_manifest["splits"]["train"]["source_split"])
    validation_split = str(curriculum_manifest["splits"]["validation"]["source_split"])
    inspection_radius = int(curriculum_manifest["inspection_radius"])

    train_scenarios = load_scenarios(
        args.dataset_dir,
        train_split,
        verify_hash=True,
    )
    validation_scenarios = load_scenarios(
        args.dataset_dir,
        validation_split,
        verify_hash=True,
    )
    scenario_tiers = scenario_tiers_from_manifest(
        train_scenarios,
        curriculum_manifest,
    )
    feasible_validation_scenarios = feasible_validation_scenarios_from_manifest(
        validation_scenarios,
        curriculum_manifest,
    )

    if args.smoke_test:
        effective_updates = 3
        easy_updates = 1
        medium_updates = 1
        validation_limit = min(
            5,
            len(feasible_validation_scenarios),
        )
        feasible_final_limit = validation_limit
        full_final_limit = min(
            5,
            len(validation_scenarios),
        )
        evaluation_frequency = 1
        checkpoint_frequency = 1
    else:
        effective_updates = args.updates
        easy_updates = args.easy_updates
        medium_updates = args.medium_updates
        validation_limit = min(
            args.validation_limit,
            len(feasible_validation_scenarios),
        )
        feasible_final_limit = (
            len(feasible_validation_scenarios)
            if args.final_validation_limit is None
            else min(
                args.final_validation_limit,
                len(feasible_validation_scenarios),
            )
        )
        full_final_limit = (
            len(validation_scenarios)
            if args.full_validation_limit is None
            else min(
                args.full_validation_limit,
                len(validation_scenarios),
            )
        )
        evaluation_frequency = args.eval_every_updates
        checkpoint_frequency = args.checkpoint_every_updates

    rollout_size = args.n_steps * args.n_envs
    total_timesteps = rollout_size * effective_updates

    run_name = args.run_name or (f"dual_critic_maskable_ppo_seed{args.seed}_u{effective_updates}")
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
        directory.mkdir(
            parents=True,
            exist_ok=True,
        )

    environment = DummyVecEnv(
        [
            build_env(
                scenario_tiers,
                inspection_radius=(inspection_radius),
            )
            for _ in range(args.n_envs)
        ]
    )
    environment.seed(args.seed)

    model = create_model(
        args,
        environment,
        tensorboard_directory,
    )

    tier_counts = {tier: len(scenarios) for tier, scenarios in scenario_tiers.items()}
    metadata: dict[str, Any] = {
        "run_name": run_name,
        "created_at_utc": (datetime.now(UTC).isoformat()),
        "git_commit": get_git_commit(),
        "torch_version": torch.__version__,
        "cuda_available": (torch.cuda.is_available()),
        "cuda_runtime": torch.version.cuda,
        "gpu_name": (torch.cuda.get_device_name(0) if torch.cuda.is_available() else None),
        "seed": args.seed,
        "algorithm": ("dual_critic_maskable_ppo"),
        "action_masking": True,
        "training_shield": False,
        "scalar_reward_penalty": False,
        "reward_critic": True,
        "cost_critic": True,
        "curriculum_manifest": str(args.curriculum_manifest),
        "curriculum_manifest_sha256": (sha256_file(args.curriculum_manifest)),
        "scenario_manifest_sha256": (sha256_file(source_manifest_path)),
        "inspection_radius": (inspection_radius),
        "train_split": train_split,
        "validation_split": validation_split,
        "train_tier_counts": tier_counts,
        "train_scenario_count": sum(tier_counts.values()),
        "validation_scenario_count": (validation_limit),
        "final_feasible_validation_count": (feasible_final_limit),
        "final_full_validation_count": (full_final_limit),
        "curriculum_schedule": {
            "easy_updates": easy_updates,
            "medium_updates": medium_updates,
            "full_updates": (effective_updates - easy_updates - medium_updates),
        },
        "n_envs": args.n_envs,
        "n_steps": args.n_steps,
        "rollout_size": rollout_size,
        "rollout_updates": effective_updates,
        "n_epochs": args.n_epochs,
        "batch_size": args.batch_size,
        "total_timesteps": total_timesteps,
        "cost_limit": args.cost_limit,
        "cost_gamma": args.cost_gamma,
        "cost_gae_lambda": (args.cost_gae_lambda),
        "cost_learning_rate": (args.cost_learning_rate),
        "dual_learning_rate": (args.dual_learning_rate),
        "dual_maximum": (args.dual_maximum),
        "dual_ema_beta": (args.dual_ema_beta),
        "dual_warmup_updates": (args.dual_warmup_updates),
        "target_kl": args.target_kl,
        "policy_init_from": (
            str(args.policy_init_from) if args.policy_init_from is not None else None
        ),
        "selection_metric": (args.selection_metric),
        "smoke_test": args.smoke_test,
        "calibration_run": (args.calibration_run),
    }

    (run_directory / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n",
        encoding="utf-8",
    )

    callbacks = CallbackList(
        [
            CurriculumProgressCallback(
                diagnostics_directory / "curriculum_progress.jsonl",
                easy_updates=easy_updates,
                medium_updates=medium_updates,
                verbose=1,
            ),
            CheckpointCallback(
                save_freq=max(
                    checkpoint_frequency * args.n_steps,
                    1,
                ),
                save_path=str(checkpoint_directory),
                name_prefix=run_name,
                verbose=2,
            ),
            ActionDiagnosticsCallback(
                diagnostics_directory / "rollout_diagnostics.jsonl",
                collapse_threshold=0.95,
                collapse_patience=3,
                verbose=1,
            ),
            DualCriticDiagnosticsCallback(
                diagnostics_directory / "dual_critic_diagnostics.jsonl",
                verbose=1,
            ),
            ScenarioEvaluationCallback(
                feasible_validation_scenarios[:validation_limit],
                eval_freq=max(
                    evaluation_frequency * args.n_steps,
                    1,
                ),
                output_directory=(evaluation_directory),
                selection_metric=(args.selection_metric),
                safety_cost_limit=(args.cost_limit),
                use_shield=False,
                use_action_masks=True,
                evaluate_at_start=True,
                verbose=1,
            ),
        ]
    )

    print(json.dumps(metadata, indent=2))

    try:
        model.learn(
            total_timesteps=total_timesteps,
            callback=callbacks,
            tb_log_name=run_name,
            reset_num_timesteps=True,
            progress_bar=args.progress_bar,
        )

        final_model_path = run_directory / "final_model"
        model.save(final_model_path)

        save_final_evaluation(
            model=model,
            scenarios=(feasible_validation_scenarios[:feasible_final_limit]),
            use_shield=False,
            output_directory=(evaluation_directory),
            output_name=("final_feasible_validation"),
        )
        save_final_evaluation(
            model=model,
            scenarios=validation_scenarios[:full_final_limit],
            use_shield=False,
            output_directory=(evaluation_directory),
            output_name=("final_full_validation"),
        )
        save_final_evaluation(
            model=model,
            scenarios=(feasible_validation_scenarios[:feasible_final_limit]),
            use_shield=True,
            output_directory=(evaluation_directory),
            output_name=("final_shielded_feasible_validation"),
        )
        save_final_evaluation(
            model=model,
            scenarios=validation_scenarios[:full_final_limit],
            use_shield=True,
            output_directory=(evaluation_directory),
            output_name=("final_shielded_full_validation"),
        )
    finally:
        environment.close()

    print(f"Final model: {final_model_path}.zip")
    print(f"Best model: {evaluation_directory / 'best_model' / 'best_model'}.zip")
    print(f"Diagnostics: {diagnostics_directory / 'dual_critic_diagnostics.jsonl'}")


if __name__ == "__main__":
    main()
