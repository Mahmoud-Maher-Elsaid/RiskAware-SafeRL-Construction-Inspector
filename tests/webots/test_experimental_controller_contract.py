from pathlib import Path

CONTROLLER = (
    Path(__file__).parents[2]
    / "webots/controllers/hierarchical_experimental_robot/hierarchical_experimental_robot.py"
)


def test_experimental_controller_uses_runtime_observation_contract() -> None:
    source = CONTROLLER.read_text(encoding="utf-8")
    assert "def observation(" not in source
    assert "causal_option_mask(" in source
    assert "structured_state(" in source
    assert "camera.getImage()" in source
    assert "target_memory" in source
    assert "recurrent_trace.jsonl" in source
    assert "torch.zeros" not in source


def test_experimental_controller_has_meaningful_decision_default() -> None:
    source = CONTROLLER.read_text(encoding="utf-8")
    assert 'RISK_AWARE_EXPERIMENTAL_DECISIONS", "100"' in source
