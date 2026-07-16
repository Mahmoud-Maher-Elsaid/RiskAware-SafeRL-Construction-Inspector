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
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from riskaware_saferrl.callbacks.action_diagnostics import (
    ActionDiagnosticsCallback,
)
from riskaware_saferrl.callbacks.curriculum_progress import (
    CurriculumProgressCallback,
)
from riskaware_saferrl.callbacks.lagrangian_update import (
    LagrangianUpdateCallback,
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
from riskaware_saferrl.models import SemanticMapExtractor
from riskaware_saferrl.safety import (
    CounterfactualLagrangianReward,
    LagrangeMultiplier,
    SafetyShield,
)
from riskaware_saferrl.scenario_dataset import load_scenarios


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
        "--algorithm",
        choices=["ppo", "maskable_ppo"],
        default="ppo",
    )
    parser.add_argument("--updates", type=int, default=100)
    parser.add_argument("--easy-updates", type=int, default=25)
    parser.add_argument("--medium-updates", type=int, default=25)
    parser.add_argument("--n-steps", type=int, default=256)
    parser.add_argument("--n-epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--timesteps", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-envs", type=int, default=4)
    parser.add_argument(
        "--device",
        choices=["cpu", "cuda", "auto"],
        default="auto",
    )
    parser.add_argument("--shield", action="store_true")
    parser.add_argument("--lagrangian", action="store_true")
    parser.add_argument(
        "--lagrange-initial-value",
        type=float,
        default=0.0,
    )
    parser.add_argument(
        "--lagrange-learning-rate",
        type=float,
        default=0.01,
    )
    parser.add_argument(
        "--lagrange-maximum",
        type=float,
        default=100.0,
    )
    parser.add_argument(
        "--proposed-cost-limit",
        type=float,
        default=5.0,
    )
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--resume-from", type=Path, default=None)
    parser.add_argument("--resume-completed-updates", type=int, default=0)
    parser.add_argument("--eval-every-updates", type=int, default=10)
    parser.add_argument("--checkpoint-every-updates", type=int, default=10)
    parser.add_argument("--validation-limit", type=int, default=50)
    parser.add_argument("--final-validation-limit", type=int, default=None)
    parser.add_argument("--full-validation-limit", type=int, default=None)
    parser.add_argument(
        "--selection-metric",
        choices=["reward", "hazard_recall", "safe_hazard_recall"],
        default="safe_hazard_recall",
    )
    parser.add_argument("--safety-cost-limit", type=float, default=5.0)
    parser.add_argument("--progress-bar", action="store_true")
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument(
        "--calibration-run",
        action="store_true",
    )
    return parser.parse_args()


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
    if args.smoke_test and args.calibration_run:
        raise ValueError("--smoke-test and --calibration-run cannot be used together.")
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
    if args.easy_updates < 1:
        raise ValueError("--easy-updates must be positive.")
    if args.medium_updates < 1:
        raise ValueError("--medium-updates must be positive.")
    if args.resume_completed_updates < 0:
        raise ValueError("--resume-completed-updates must be non-negative.")
    if args.resume_from is None and args.resume_completed_updates:
        raise ValueError("--resume-completed-updates requires --resume-from.")

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
    if args.lagrangian and not args.shield:
        raise ValueError("--lagrangian requires --shield.")
    if args.lagrangian and args.algorithm != "maskable_ppo":
        raise ValueError("--lagrangian requires --algorithm maskable_ppo.")
    if args.lagrange_initial_value < 0.0:
        raise ValueError("--lagrange-initial-value must be non-negative.")
    if args.lagrange_learning_rate <= 0.0:
        raise ValueError("--lagrange-learning-rate must be positive.")
    if args.lagrange_maximum <= 0.0:
        raise ValueError("--lagrange-maximum must be positive.")
    if args.proposed_cost_limit < 0.0:
        raise ValueError("--proposed-cost-limit must be non-negative.")

    if not args.calibration_run and (not args.smoke_test):
        total_scheduled_updates = args.resume_completed_updates + args.updates
        if args.calibration_run and total_scheduled_updates < 3:
            raise ValueError("Calibration requires at least 3 total rollout updates.")
        if total_scheduled_updates < 100:
            raise ValueError(
                "Full curriculum training requires at least 100 total rollout updates."
            )
        if args.easy_updates + args.medium_updates >= total_scheduled_updates:
            raise ValueError("The curriculum schedule must leave at least one full-stage update.")


def build_env(
    scenario_tiers,
    *,
    inspection_radius: int,
    use_shield: bool,
    use_lagrangian: bool,
    lagrange_multiplier: LagrangeMultiplier | None,
):
    def create_environment():
        environment = CurriculumConstructionInspectionEnv(
            scenario_tiers,
            inspection_radius=inspection_radius,
        )
        if use_shield:
            environment = SafetyShield(environment)
        if use_lagrangian:
            if lagrange_multiplier is None:
                raise RuntimeError("Lagrangian training requires a shared multiplier.")
            environment = CounterfactualLagrangianReward(
                environment,
                lagrange_multiplier,
            )
        return Monitor(environment)

    return create_environment


def create_model(
    args: argparse.Namespace,
    environment: DummyVecEnv,
    tensorboard_directory: Path,
) -> PPO | MaskablePPO:
    algorithm_class = MaskablePPO if args.algorithm == "maskable_ppo" else PPO

    if args.resume_from is not None:
        model = algorithm_class.load(
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

    return algorithm_class(
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

    source_manifest_path = args.dataset_dir / "manifest.json"
    curriculum_manifest = load_curriculum_manifest(
        args.curriculum_manifest,
        source_manifest_path=source_manifest_path,
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
        validation_limit = min(5, len(feasible_validation_scenarios))
        final_validation_limit = validation_limit
        full_validation_limit = min(5, len(validation_scenarios))
        eval_every_updates = 1
        checkpoint_every_updates = 1
    else:
        effective_updates = args.updates
        easy_updates = args.easy_updates
        medium_updates = args.medium_updates
        validation_limit = min(
            args.validation_limit,
            len(feasible_validation_scenarios),
        )
        final_validation_limit = (
            len(feasible_validation_scenarios)
            if args.final_validation_limit is None
            else min(
                args.final_validation_limit,
                len(feasible_validation_scenarios),
            )
        )
        full_validation_limit = (
            len(validation_scenarios)
            if args.full_validation_limit is None
            else min(args.full_validation_limit, len(validation_scenarios))
        )
        eval_every_updates = args.eval_every_updates
        checkpoint_every_updates = args.checkpoint_every_updates

    rollout_size = args.n_steps * args.n_envs
    total_timesteps = (
        args.timesteps if args.timesteps is not None else rollout_size * effective_updates
    )

    if not args.smoke_test and total_timesteps < rollout_size * args.updates:
        raise ValueError("--timesteps cannot be lower than the requested rollout updates.")

    use_action_masks = args.algorithm == "maskable_ppo"
    lagrange_multiplier = (
        LagrangeMultiplier(
            value=args.lagrange_initial_value,
            learning_rate=args.lagrange_learning_rate,
            maximum=args.lagrange_maximum,
        )
        if args.lagrangian
        else None
    )
    periodic_evaluation_shield = args.shield and not args.lagrangian

    if args.run_name is not None:
        run_name = args.run_name
    else:
        shield_suffix = "_shielded" if args.shield else ""
        run_name = f"{args.algorithm}_safe_curriculum{shield_suffix}_seed{args.seed}"

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

    environment = DummyVecEnv(
        [
            build_env(
                scenario_tiers,
                inspection_radius=inspection_radius,
                use_shield=args.shield,
                use_lagrangian=args.lagrangian,
                lagrange_multiplier=lagrange_multiplier,
            )
            for _ in range(args.n_envs)
        ]
    )
    environment.seed(args.seed)
    model = create_model(args, environment, tensorboard_directory)

    tier_counts = {tier: len(scenarios) for tier, scenarios in scenario_tiers.items()}
    metadata: dict[str, Any] = {
        "run_name": run_name,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "git_commit": get_git_commit(),
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_runtime": torch.version.cuda,
        "gpu_name": (torch.cuda.get_device_name(0) if torch.cuda.is_available() else None),
        "seed": args.seed,
        "curriculum_manifest": str(args.curriculum_manifest),
        "curriculum_manifest_sha256": sha256_file(args.curriculum_manifest),
        "scenario_manifest_sha256": sha256_file(source_manifest_path),
        "inspection_radius": inspection_radius,
        "train_split": train_split,
        "validation_split": validation_split,
        "train_tier_counts": tier_counts,
        "train_scenario_count": sum(tier_counts.values()),
        "validation_scenario_count": validation_limit,
        "final_feasible_validation_count": final_validation_limit,
        "final_full_validation_count": full_validation_limit,
        "curriculum_schedule": {
            "easy_updates": easy_updates,
            "medium_updates": medium_updates,
            "full_updates": max(
                0,
                args.resume_completed_updates + effective_updates - easy_updates - medium_updates,
            ),
            "initial_completed_updates": args.resume_completed_updates,
        },
        "n_envs": args.n_envs,
        "n_steps": args.n_steps,
        "rollout_size": rollout_size,
        "rollout_updates": effective_updates,
        "n_epochs": args.n_epochs,
        "total_optimization_epochs": effective_updates * args.n_epochs,
        "batch_size": args.batch_size,
        "total_timesteps": total_timesteps,
        "algorithm": args.algorithm,
        "action_masking": use_action_masks,
        "shield": args.shield,
        "lagrangian": args.lagrangian,
        "periodic_evaluation_shield": (periodic_evaluation_shield),
        "lagrange_initial_value": (args.lagrange_initial_value),
        "lagrange_learning_rate": (args.lagrange_learning_rate),
        "lagrange_maximum": args.lagrange_maximum,
        "proposed_cost_limit": (args.proposed_cost_limit),
        "selection_metric": args.selection_metric,
        "safety_cost_limit": args.safety_cost_limit,
        "resume_from": str(args.resume_from) if args.resume_from else None,
        "resume_completed_updates": args.resume_completed_updates,
        "smoke_test": args.smoke_test,
        "calibration_run": args.calibration_run,
    }
    (run_directory / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n",
        encoding="utf-8",
    )

    callback_items = [
        CurriculumProgressCallback(
            diagnostics_directory / "curriculum_progress.jsonl",
            easy_updates=easy_updates,
            medium_updates=medium_updates,
            initial_completed_updates=args.resume_completed_updates,
            verbose=1,
        ),
        CheckpointCallback(
            save_freq=max(
                checkpoint_every_updates * args.n_steps,
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
        ScenarioEvaluationCallback(
            feasible_validation_scenarios[:validation_limit],
            eval_freq=max(
                eval_every_updates * args.n_steps,
                1,
            ),
            output_directory=evaluation_directory,
            selection_metric=args.selection_metric,
            safety_cost_limit=args.safety_cost_limit,
            use_shield=periodic_evaluation_shield,
            use_action_masks=use_action_masks,
            evaluate_at_start=True,
            verbose=1,
        ),
    ]

    if lagrange_multiplier is not None:
        callback_items.insert(
            3,
            LagrangianUpdateCallback(
                lagrange_multiplier,
                cost_limit=args.proposed_cost_limit,
                output_path=(diagnostics_directory / "lagrangian_diagnostics.jsonl"),
                verbose=1,
            ),
        )

    callbacks = CallbackList(callback_items)
    print(json.dumps(metadata, indent=2))

    try:
        model.learn(
            total_timesteps=total_timesteps,
            callback=callbacks,
            tb_log_name=run_name,
            reset_num_timesteps=args.resume_from is None,
            progress_bar=args.progress_bar,
        )

        final_model_path = run_directory / "final_model"
        model.save(final_model_path)

        primary_evaluation_shield = args.shield and not args.lagrangian
        feasible_records, feasible_summary = evaluate_policy_on_scenarios(
            model,
            feasible_validation_scenarios[:final_validation_limit],
            use_shield=primary_evaluation_shield,
            use_action_masks=use_action_masks,
            deterministic=True,
        )
        feasible_summary["timesteps"] = int(model.num_timesteps)
        feasible_summary["evaluation_type"] = "final_feasible_validation"
        save_evaluation_results(
            feasible_records,
            feasible_summary,
            evaluation_directory,
            "final_feasible_validation",
        )

        full_records, full_summary = evaluate_policy_on_scenarios(
            model,
            validation_scenarios[:full_validation_limit],
            use_shield=primary_evaluation_shield,
            use_action_masks=use_action_masks,
            deterministic=True,
        )
        full_summary["timesteps"] = int(model.num_timesteps)
        full_summary["evaluation_type"] = "final_full_validation"
        save_evaluation_results(
            full_records,
            full_summary,
            evaluation_directory,
            "final_full_validation",
        )

        if args.lagrangian and args.shield:
            (
                shielded_feasible_records,
                shielded_feasible_summary,
            ) = evaluate_policy_on_scenarios(
                model,
                feasible_validation_scenarios[:final_validation_limit],
                use_shield=True,
                use_action_masks=use_action_masks,
                deterministic=True,
            )
            shielded_feasible_summary["timesteps"] = int(model.num_timesteps)
            shielded_feasible_summary["evaluation_type"] = "final_shielded_feasible_validation"
            save_evaluation_results(
                shielded_feasible_records,
                shielded_feasible_summary,
                evaluation_directory,
                "final_shielded_feasible_validation",
            )

            (
                shielded_full_records,
                shielded_full_summary,
            ) = evaluate_policy_on_scenarios(
                model,
                validation_scenarios[:full_validation_limit],
                use_shield=True,
                use_action_masks=use_action_masks,
                deterministic=True,
            )
            shielded_full_summary["timesteps"] = int(model.num_timesteps)
            shielded_full_summary["evaluation_type"] = "final_shielded_full_validation"
            save_evaluation_results(
                shielded_full_records,
                shielded_full_summary,
                evaluation_directory,
                "final_shielded_full_validation",
            )
    finally:
        environment.close()

    print(f"Final model: {final_model_path}.zip")
    print(f"Best model: {evaluation_directory / 'best_model' / 'best_model'}.zip")
    print(f"Diagnostics: {diagnostics_directory / 'rollout_diagnostics.jsonl'}")


if __name__ == "__main__":
    main()
