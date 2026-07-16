import numpy as np

from riskaware_saferrl.callbacks import ActionDiagnosticsCallback


def test_action_diagnostics_counts_actions(tmp_path) -> None:
    callback = ActionDiagnosticsCallback(tmp_path / "diagnostics.jsonl")
    callback.locals = {
        "actions": np.array([0, 4, 4, 2]),
        "infos": [
            {"cost": 0.0},
            {"cost": 1.0, "cost_collision": 1.0},
            {"cost": 0.0},
            {"cost": 0.5, "cost_worker": 0.5},
        ],
        "dones": np.array([False, True, False, True]),
    }

    callback._on_rollout_start()
    assert callback._on_step() is True
    assert callback.action_counts.tolist() == [1, 0, 1, 0, 2]
    assert callback.transition_count == 4
    assert callback.cost_sums["safety_cost"] == 1.5
    assert len(callback.terminal_records) == 2
