from __future__ import annotations

import hashlib
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
import yaml
from stable_baselines3 import PPO, SAC

from riskaware_saferrl.baselines import (
    FrontierExplorationPlanner,
    RiskAwareAStarPlanner,
)
from riskaware_saferrl.envs import GridEnvironmentConfig, ResearchConstructionEnv
from riskaware_saferrl.safety import PredictiveSafetyShield, PredictiveShieldWrapper

PLANNER_ALGORITHMS = {"risk_aware_astar", "frontier_exploration"}
POLICY_ALGORITHMS = {"PPO", "SAC", "RiskShield-PPO"}
REQUIRED_RESULT_FIELDS = {
    "run_id",
    "configuration_hash",
    "environment_size",
    "hazard_density",
    "perception_noise",
    "algorithm",
    "evaluation_seed",
    "checkpoint_path",
    "checkpoint_sha256",
    "hazard_recall",
    "inspection_coverage",
    "collision_rate",
    "collision_count",
    "near_miss_rate",
    "near_miss_count",
    "constraint_violations",
    "restricted_zone_violations",
    "time_to_inspect",
    "mission_duration",
    "energy_usage",
    "robustness_score",
    "success",
    "safety_cost",
    "path_length",
    "shield_interventions",
    "emergency_stops",
    "inference_latency_ms",
    "policy_latency_ms",
    "episode_steps",
    "terminated",
    "truncated",
    "failure_reason",
    "started_at",
    "completed_at",
}


@dataclass(frozen=True)
class BenchmarkRun:
    run_id: str
    environment_size_name: str
    environment_size: int
    hazard_density_name: str
    hazard_density: float
    perception_noise: float
    algorithm: str
    evaluation_seed: int
    checkpoint_path: str
    checkpoint_sha256: str
    shield_mode: str
    benchmark_configuration_hash: str
    configuration_hash: str

    @property
    def seed(self) -> int:
        return self.evaluation_seed


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def stable_hash(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def load_settings(config_path: Path) -> dict[str, Any]:
    settings = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    required = {
        "environment_sizes",
        "hazard_densities",
        "perception_noise_levels",
        "algorithms",
        "evaluation_seeds",
        "expected_primary_runs",
        "checkpoints",
    }
    missing = sorted(required - settings.keys())
    if missing:
        raise ValueError(f"Benchmark configuration is missing: {missing}")
    if set(settings["algorithms"]) != PLANNER_ALGORITHMS | POLICY_ALGORITHMS:
        raise ValueError("Benchmark must configure exactly the five accepted algorithms")
    return settings


def _checkpoint_metadata(settings: dict[str, Any], algorithm: str) -> tuple[str, str]:
    if algorithm in PLANNER_ALGORITHMS:
        return "", ""
    configured = Path(settings["checkpoints"][algorithm])
    if not configured.is_file():
        raise FileNotFoundError(f"Checkpoint is missing: {configured}")
    return configured.as_posix(), file_sha256(configured)


def build_manifest(settings: dict[str, Any]) -> list[BenchmarkRun]:
    benchmark_hash = stable_hash(settings)
    checkpoint_metadata = {
        algorithm: _checkpoint_metadata(settings, algorithm) for algorithm in settings["algorithms"]
    }
    runs = []
    for size_name, size in settings["environment_sizes"].items():
        for density_name, density in settings["hazard_densities"].items():
            for noise in settings["perception_noise_levels"]:
                for algorithm in settings["algorithms"]:
                    checkpoint_path, checkpoint_hash = checkpoint_metadata[algorithm]
                    shield_mode = "k_step" if algorithm == "RiskShield-PPO" else "disabled"
                    for seed in settings["evaluation_seeds"]:
                        payload = {
                            "environment_size": size_name,
                            "hazard_density": density_name,
                            "perception_noise": float(noise),
                            "algorithm": algorithm,
                            "evaluation_seed": int(seed),
                            "checkpoint_sha256": checkpoint_hash,
                            "shield_mode": shield_mode,
                            "benchmark_configuration_hash": benchmark_hash,
                        }
                        configuration_hash = stable_hash(payload)
                        slug = algorithm.lower().replace("-", "_")
                        runs.append(
                            BenchmarkRun(
                                run_id=(
                                    f"{size_name}-{density_name}-n{float(noise):.1f}-"
                                    f"{slug}-s{seed}-{configuration_hash[:12]}"
                                ),
                                environment_size_name=size_name,
                                environment_size=int(size),
                                hazard_density_name=density_name,
                                hazard_density=float(density),
                                perception_noise=float(noise),
                                algorithm=algorithm,
                                evaluation_seed=int(seed),
                                checkpoint_path=checkpoint_path,
                                checkpoint_sha256=checkpoint_hash,
                                shield_mode=shield_mode,
                                benchmark_configuration_hash=benchmark_hash,
                                configuration_hash=configuration_hash,
                            )
                        )
    run_ids = [run.run_id for run in runs]
    hashes = [run.configuration_hash for run in runs]
    if len(run_ids) != len(set(run_ids)) or len(hashes) != len(set(hashes)):
        raise RuntimeError("Benchmark manifest contains duplicate identifiers")
    expected = int(settings["expected_primary_runs"])
    if len(runs) != expected:
        raise RuntimeError(f"Expected {expected} runs, generated {len(runs)}")
    return runs


def grid_config(run: BenchmarkRun) -> GridEnvironmentConfig:
    dynamic_density = {"low": 0.01, "medium": 0.03, "high": 0.05}[run.hazard_density_name]
    return GridEnvironmentConfig(
        size=run.environment_size,
        obstacle_density=0.10 + 0.02 * (run.environment_size / 16),
        hazard_density=run.hazard_density,
        worker_density=0.025,
        restricted_density=0.04,
        dynamic_hazard_density=dynamic_density,
        ppe_risk_density=0.025,
        perception_false_negative_rate=run.perception_noise,
        max_steps=run.environment_size * 22,
    )


def base_metrics(
    run: BenchmarkRun,
    environment: ResearchConstructionEnv,
    info: dict[str, Any],
    *,
    started_at: str,
    duration: float,
    inference_latency: float,
    policy_latency: float,
    shield_interventions: int,
    emergency_stops: int,
    terminated: bool,
    truncated: bool,
) -> dict[str, Any]:
    telemetry = environment.telemetry()
    safety_cost = float(telemetry.safety_cost)
    robustness = float(
        np.clip(
            0.4 * float(info["hazard_recall"])
            + 0.3 * float(info["coverage"])
            + 0.3 / (1.0 + safety_cost),
            0.0,
            1.0,
        )
    )
    result = {
        **asdict(run),
        "hazard_recall": float(info["hazard_recall"]),
        "inspection_coverage": float(info["coverage"]),
        "collision_rate": float(telemetry.collisions > 0),
        "collision_count": int(telemetry.collisions),
        "near_miss_rate": float(telemetry.near_misses / max(1, telemetry.steps)),
        "near_miss_count": int(telemetry.near_misses),
        "constraint_violations": int(telemetry.near_misses + telemetry.restricted_violations),
        "restricted_zone_violations": int(telemetry.restricted_violations),
        "time_to_inspect": int(telemetry.steps),
        "mission_duration": float(duration),
        "energy_usage": float(telemetry.energy_usage),
        "robustness_score": robustness,
        "success": bool(info["success"]),
        "success_rate": float(info["success"]),
        "safety_cost": safety_cost,
        "path_length": float(telemetry.path_length),
        "shield_interventions": int(shield_interventions),
        "emergency_stops": int(emergency_stops),
        "inference_latency_ms": float(inference_latency),
        "policy_latency_ms": float(policy_latency),
        "episode_steps": int(telemetry.steps),
        "terminated": bool(terminated),
        "truncated": bool(truncated),
        "failure_reason": "",
        "started_at": started_at,
        "completed_at": utc_now(),
        "completed": True,
    }
    missing = REQUIRED_RESULT_FIELDS - result.keys()
    if missing:
        raise RuntimeError(f"Benchmark record schema is incomplete: {sorted(missing)}")
    return result


def evaluate_planner(run: BenchmarkRun) -> dict[str, Any]:
    started_at = utc_now()
    environment = ResearchConstructionEnv(grid_config(run))
    planner = (
        RiskAwareAStarPlanner()
        if run.algorithm == "risk_aware_astar"
        else FrontierExplorationPlanner()
    )
    try:
        _, info = environment.reset(seed=run.seed)
        planner.reset()
        terminated = truncated = False
        latencies = []
        started_episode = time.perf_counter()
        while not (terminated or truncated):
            started = time.perf_counter()
            decision = planner.decide(environment)
            latencies.append((time.perf_counter() - started) * 1000)
            _, _, terminated, truncated, info = environment.step(decision.action)
        return base_metrics(
            run,
            environment,
            info,
            started_at=started_at,
            duration=time.perf_counter() - started_episode,
            inference_latency=0.0,
            policy_latency=float(np.mean(latencies)),
            shield_interventions=0,
            emergency_stops=0,
            terminated=terminated,
            truncated=truncated,
        )
    finally:
        environment.close()


def evaluate_policy(run: BenchmarkRun, model: Any, horizon: int) -> dict[str, Any]:
    started_at = utc_now()
    base = ResearchConstructionEnv(grid_config(run))
    continuous = run.algorithm == "SAC"
    environment: gym.Env = base
    if run.algorithm == "RiskShield-PPO":
        environment = PredictiveShieldWrapper(base, PredictiveSafetyShield(horizon=horizon))
    if continuous:
        from riskaware_saferrl.training import ContinuousActionAdapter

        environment = ContinuousActionAdapter(environment)
    environment = gym.wrappers.FlattenObservation(environment)
    try:
        observation, info = environment.reset(seed=run.seed)
        terminated = truncated = False
        policy_latencies = []
        shield_interventions = 0
        emergency_stops = 0
        started_episode = time.perf_counter()
        while not (terminated or truncated):
            started = time.perf_counter()
            action, _ = model.predict(observation, deterministic=True)
            policy_latencies.append((time.perf_counter() - started) * 1000)
            if not continuous:
                action = int(np.asarray(action).reshape(-1)[0])
            observation, _, terminated, truncated, info = environment.step(action)
            shield_interventions += int(bool(info.get("shield_intervention", False)))
            emergency_stops += int(bool(info.get("emergency_stop", False)))
        return base_metrics(
            run,
            base,
            info,
            started_at=started_at,
            duration=time.perf_counter() - started_episode,
            inference_latency=0.0,
            policy_latency=float(np.mean(policy_latencies)),
            shield_interventions=shield_interventions,
            emergency_stops=emergency_stops,
            terminated=terminated,
            truncated=truncated,
        )
    finally:
        environment.close()


def _valid_cached_record(path: Path, run: BenchmarkRun) -> dict[str, Any] | None:
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if (
        record.get("run_id") != run.run_id
        or record.get("configuration_hash") != run.configuration_hash
        or not record.get("completed")
        or REQUIRED_RESULT_FIELDS - record.keys()
    ):
        return None
    return record


def _write_resume(
    output_directory: Path,
    completed: int,
    expected: int,
    failed: int,
    last_run_id: str,
) -> None:
    atomic_write_json(
        output_directory / "resume_state.json",
        {
            "completed": completed,
            "expected": expected,
            "missing": expected - completed,
            "failed": failed,
            "last_run_id": last_run_id,
            "updated_at": utc_now(),
        },
    )


def run_benchmark(
    config_path: Path,
    output_directory: Path,
    *,
    limit: int | None = None,
    one_per_algorithm: bool = False,
) -> list[dict[str, Any]]:
    settings = load_settings(config_path)
    manifest = build_manifest(settings)
    if one_per_algorithm:
        first_by_algorithm: dict[str, BenchmarkRun] = {}
        for run in manifest:
            first_by_algorithm.setdefault(run.algorithm, run)
        manifest = [first_by_algorithm[name] for name in settings["algorithms"]]
    elif limit is not None:
        manifest = manifest[:limit]
    output_directory.mkdir(parents=True, exist_ok=True)
    cache = output_directory / "cache"
    cache.mkdir(parents=True, exist_ok=True)
    completed: dict[str, dict[str, Any]] = {}
    pending = []
    for run in manifest:
        cache_path = cache / f"{run.configuration_hash}.json"
        record = _valid_cached_record(cache_path, run) if cache_path.exists() else None
        if record is not None:
            completed[run.run_id] = record
        else:
            pending.append(run)

    failures: list[dict[str, Any]] = []
    timeout_seconds = float(settings.get("episode_timeout_seconds", 0))

    def persist(run: BenchmarkRun, record: dict[str, Any], elapsed: float) -> None:
        if timeout_seconds > 0 and elapsed > timeout_seconds:
            raise TimeoutError(f"{run.run_id} exceeded {timeout_seconds:.1f}s ({elapsed:.3f}s)")
        atomic_write_json(cache / f"{run.configuration_hash}.json", record)
        completed[run.run_id] = record
        _write_resume(
            output_directory,
            len(completed),
            len(manifest),
            len(failures),
            run.run_id,
        )
        print(
            f"BENCHMARK_PROGRESS={len(completed)}/{len(manifest)} "
            f"remaining={len(manifest) - len(completed)} run_id={run.run_id}",
            flush=True,
        )

    planner_runs = [run for run in pending if run.algorithm in PLANNER_ALGORITHMS]
    with ThreadPoolExecutor(max_workers=int(settings["planner_workers"])) as executor:
        futures = {executor.submit(evaluate_planner, run): run for run in planner_runs}
        for future in as_completed(futures):
            run = futures[future]
            try:
                record = future.result()
                persist(run, record, float(record["mission_duration"]))
            except Exception as exc:
                failures.append(
                    {
                        "run_id": run.run_id,
                        "failure_reason": f"{type(exc).__name__}: {exc}",
                        "recorded_at": utc_now(),
                    }
                )

    models: dict[str, Any] = {}
    for run in pending:
        if run.algorithm in PLANNER_ALGORITHMS:
            continue
        try:
            if run.algorithm not in models:
                model_class = SAC if run.algorithm == "SAC" else PPO
                models[run.algorithm] = model_class.load(run.checkpoint_path, device="cuda")
            started = time.perf_counter()
            record = evaluate_policy(
                run,
                models[run.algorithm],
                int(settings["predictive_shield_horizon"]),
            )
            persist(run, record, time.perf_counter() - started)
        except Exception as exc:
            failures.append(
                {
                    "run_id": run.run_id,
                    "failure_reason": f"{type(exc).__name__}: {exc}",
                    "recorded_at": utc_now(),
                }
            )
            _write_resume(
                output_directory,
                len(completed),
                len(manifest),
                len(failures),
                run.run_id,
            )

    failed_path = output_directory / "failed_runs.jsonl"
    failed_path.write_text(
        "".join(json.dumps(failure) + "\n" for failure in failures),
        encoding="utf-8",
    )
    records = [completed[run.run_id] for run in manifest if run.run_id in completed]
    if failures or len(records) != len(manifest):
        raise RuntimeError(
            f"Benchmark incomplete: {len(records)}/{len(manifest)}, "
            f"{len(failures)} unresolved failures"
        )
    if len({record["run_id"] for record in records}) != len(records):
        raise RuntimeError("Completed benchmark contains duplicate run identifiers")
    return records
