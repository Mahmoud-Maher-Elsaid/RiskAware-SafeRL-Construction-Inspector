from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def main() -> None:
    root = Path("reports/final_submission")
    assert read_json(root / "stage1_grid_environment/validation.json")["status"] == "PASSED"
    assert len(pd.read_csv(root / "stage2_planner_baselines/raw_results.csv")) == 90
    assert read_json(root / "stage3_rl_baselines/summary.json")["status"] == "PASSED"
    assert read_json(root / "stage4_riskshield_ppo/comparison_summary.json")["status"] == "PASSED"
    assert read_json(root / "stage5_safety_shield/summary.json")["status"] == "PASSED"
    worlds = read_json(root / "stage6_webots/world_smoke_summary.json")
    assert len(worlds) == 3
    assert all(
        world["runtime_verified"] and world["mission_completed"] and world["physics_stable"]
        for world in worlds
    )
    assert read_json(root / "stage7_perception/validation.json")["status"] == "PASSED"
    assert read_json(root / "stage8_uncertainty/summary.json")["run_count"] == 320
    benchmark = read_json(root / "stage9_benchmark/benchmark_summary.json")
    assert benchmark["run_count"] == benchmark["unique_run_ids"] == 1350
    assert benchmark["missing_run_count"] == benchmark["failed_run_count"] == 0
    ablation = read_json(root / "stage9_benchmark/ablation_summary.json")
    assert ablation["run_count"] == ablation["unique_run_count"] == 180
    assert read_json(root / "paper_result.json")["status"] == "PASSED"
    print("FINAL_EVIDENCE_SCHEMA=PASSED")


if __name__ == "__main__":
    main()
