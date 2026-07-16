import numpy as np

from riskaware_saferrl.evaluation.scenario_evaluator import (
    compute_selection_score,
    evaluate_policy_on_scenarios,
)
from riskaware_saferrl.scenario_dataset import load_scenarios


class InspectOnlyPolicy:
    def predict(self, observation, deterministic=True):
        return np.array(4), None


def test_evaluator_returns_confidence_statistics() -> None:
    scenario = load_scenarios("data/scenarios", "validation")[0]
    records, summary = evaluate_policy_on_scenarios(
        InspectOnlyPolicy(),
        [scenario],
    )

    assert len(records) == 1
    assert summary["scenario_count"] == 1
    assert summary["metrics"]["reward"]["ci95"] == 0.0
    assert summary["metrics"]["hazard_recall"]["mean"] >= 0.0


def test_safe_selection_penalizes_constraint_violation() -> None:
    summary = {
        "metrics": {
            "reward": {"mean": 5.0},
            "hazard_recall": {"mean": 0.8},
            "safety_cost": {"mean": 7.0},
        }
    }
    score = compute_selection_score(
        summary,
        selection_metric="safe_hazard_recall",
        safety_cost_limit=5.0,
    )
    assert score < 0.0
