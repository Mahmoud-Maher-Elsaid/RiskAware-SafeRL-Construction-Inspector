from __future__ import annotations

import torch

from riskaware_saferrl.policies import RecurrentMaskedPolicy


def inputs(batch: int = 3, sequence: int = 4):
    maps = torch.rand(batch, sequence, 11, 16, 16)
    states = torch.rand(batch, sequence, 17)
    masks = torch.ones(batch, sequence, 5, dtype=torch.bool)
    masks[..., 4] = False
    starts = torch.zeros(batch, sequence, dtype=torch.bool)
    return maps, states, masks, starts


def test_shapes_mask_normalization_and_invalid_probability() -> None:
    policy = RecurrentMaskedPolicy()
    maps, states, masks, starts = inputs()
    output = policy(maps, states, masks, episode_starts=starts)
    assert output.recurrent_state.shape == (1, 3, 256)
    assert output.reward_value.shape == (3, 4)
    assert output.cost_value.shape == (3, 4)
    assert output.subgoal_logits.shape == (3, 4, 6)
    torch.testing.assert_close(output.distribution.probs.sum(-1), torch.ones(3, 4))
    assert torch.all(output.distribution.probs[..., 4] == 0)


def test_recurrent_state_resets_at_episode_boundary() -> None:
    torch.manual_seed(3)
    policy = RecurrentMaskedPolicy()
    maps, states, masks, starts = inputs(batch=1, sequence=3)
    starts[:, 1] = True
    combined = policy(maps, states, masks, episode_starts=starts)
    isolated = policy(maps[:, 1:], states[:, 1:], masks[:, 1:])
    torch.testing.assert_close(
        combined.distribution.probs[:, 1:], isolated.distribution.probs, atol=1e-6, rtol=1e-5
    )


def test_deterministic_inference_and_checkpoint_round_trip(tmp_path) -> None:
    torch.manual_seed(5)
    policy = RecurrentMaskedPolicy()
    maps, states, masks, _ = inputs(batch=1, sequence=1)
    action, output = policy.predict(maps, states, masks)
    path = tmp_path / "policy.pt"
    torch.save(policy.state_dict(), path)
    restored = RecurrentMaskedPolicy()
    restored.load_state_dict(torch.load(path, weights_only=True))
    restored_action, restored_output = restored.predict(maps, states, masks)
    torch.testing.assert_close(action, restored_action)
    torch.testing.assert_close(output.distribution.probs, restored_output.distribution.probs)


def test_cuda_inference_when_available() -> None:
    if not torch.cuda.is_available():
        return
    policy = RecurrentMaskedPolicy().cuda()
    maps, states, masks, _ = inputs(batch=1, sequence=1)
    action, output = policy.predict(maps.cuda(), states.cuda(), masks.cuda())
    assert action.is_cuda
    assert output.recurrent_state.is_cuda
