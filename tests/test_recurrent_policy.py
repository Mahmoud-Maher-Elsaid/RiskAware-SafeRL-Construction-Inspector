import torch

from riskaware_saferrl.models.recurrent_policy import RecurrentSemanticPolicy


def test_recurrent_policy_shapes_and_reset_are_deterministic() -> None:
    torch.manual_seed(42)
    model = RecurrentSemanticPolicy(observation_size=12, hidden_size=16)
    observations = torch.zeros(2, 4, 12)
    initial = model.initial_state(2)
    logits, values, state = model(observations, initial)
    reset_logits, _, reset_state = model(observations, model.initial_state(2))
    assert logits.shape == (2, 4, 5)
    assert values.shape == (2, 4, 1)
    assert state.shape == (1, 2, 16)
    assert torch.equal(logits, reset_logits)
    assert torch.equal(state, reset_state)


def test_recurrent_state_carries_history_without_global_state() -> None:
    model = RecurrentSemanticPolicy(observation_size=8, hidden_size=8)
    first = torch.ones(1, 1, 8)
    second = torch.zeros(1, 1, 8)
    _, _, state = model(first)
    carried_logits, _, _ = model(second, state)
    reset_logits, _, _ = model(second, model.initial_state(1))
    assert not torch.equal(carried_logits, reset_logits)
