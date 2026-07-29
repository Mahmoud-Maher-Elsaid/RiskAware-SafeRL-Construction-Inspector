from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from riskaware_saferrl.benchmarking import (
    REQUIRED_RESULT_FIELDS,
    build_manifest,
    run_benchmark,
)

CONFIG = Path("configs/benchmarks/full_matrix.yaml")


def settings() -> dict:
    return yaml.safe_load(CONFIG.read_text(encoding="utf-8"))


def test_full_manifest_has_exact_unique_matrix() -> None:
    manifest = build_manifest(settings())
    assert len(manifest) == 1350
    assert len({run.run_id for run in manifest}) == 1350
    assert len({run.configuration_hash for run in manifest}) == 1350


def test_manifest_contains_every_algorithm_and_seed() -> None:
    configured = settings()
    manifest = build_manifest(configured)
    assert {run.algorithm for run in manifest} == set(configured["algorithms"])
    assert {run.seed for run in manifest} == set(configured["evaluation_seeds"])
    assert {run.perception_noise for run in manifest} == {0.0, 0.1, 0.2}


def test_manifest_ids_include_immutable_configuration_evidence() -> None:
    manifest = build_manifest(settings())
    policy = next(run for run in manifest if run.algorithm == "PPO")
    planner = next(run for run in manifest if run.algorithm == "risk_aware_astar")
    assert policy.configuration_hash[:12] in policy.run_id
    assert len(policy.checkpoint_sha256) == 64
    assert policy.shield_mode == "disabled"
    assert planner.checkpoint_sha256 == ""


def test_smoke_record_schema_resume_and_determinism(tmp_path: Path) -> None:
    first = run_benchmark(CONFIG, tmp_path, limit=1)
    cache_files = list((tmp_path / "cache").glob("*.json"))
    assert len(first) == 1
    assert REQUIRED_RESULT_FIELDS <= first[0].keys()
    assert len(cache_files) == 1
    first_mtime = cache_files[0].stat().st_mtime_ns
    second = run_benchmark(CONFIG, tmp_path, limit=1)
    assert cache_files[0].stat().st_mtime_ns == first_mtime
    deterministic_fields = REQUIRED_RESULT_FIELDS - {
        "started_at",
        "completed_at",
        "mission_duration",
        "policy_latency_ms",
        "inference_latency_ms",
    }
    assert {key: first[0][key] for key in deterministic_fields} == {
        key: second[0][key] for key in deterministic_fields
    }
    assert not list((tmp_path / "cache").glob("*.tmp"))
    resume = json.loads((tmp_path / "resume_state.json").read_text(encoding="utf-8"))
    assert resume["completed"] == resume["expected"] == 1


def test_timeout_is_recorded_as_unresolved_failure(tmp_path: Path) -> None:
    configured = settings()
    configured["episode_timeout_seconds"] = 1e-12
    config_path = tmp_path / "timeout.yaml"
    config_path.write_text(yaml.safe_dump(configured), encoding="utf-8")
    with pytest.raises(RuntimeError, match="unresolved failures"):
        run_benchmark(config_path, tmp_path / "output", limit=1)
    failures = [
        json.loads(line)
        for line in (tmp_path / "output" / "failed_runs.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert len(failures) == 1
    assert "TimeoutError" in failures[0]["failure_reason"]
