from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import torch

from riskaware_saferrl.envs import ResearchConstructionEnvV2
from riskaware_saferrl.policies import RecurrentMaskedPolicy


def load_script():
    path = Path("scripts/run_dagger_v2.py")
    spec = importlib.util.spec_from_file_location("run_dagger_v2", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_policy_action_preserves_hidden_state_and_mask() -> None:
    module = load_script()
    environment = ResearchConstructionEnvV2()
    observation, _ = environment.reset(seed=7)
    policy = RecurrentMaskedPolicy().eval()
    action, hidden = module.policy_action(policy, observation, None, torch.device("cpu"))
    assert bool(observation["action_mask"][action])
    assert hidden.shape == (1, 1, policy.recurrent_hidden_size)
    next_observation, *_ = environment.step(action)
    second_action, second_hidden = module.policy_action(
        policy, next_observation, hidden, torch.device("cpu")
    )
    assert bool(next_observation["action_mask"][second_action])
    assert not torch.equal(hidden, second_hidden)


def test_cumulative_manifest_keeps_causal_source_and_corrections(tmp_path: Path) -> None:
    module = load_script()
    base = {
        "status": "PASSED",
        "transitions": 10,
        "chunks": [{"path": "base.npz", "rows": 10, "sha256": "a" * 64}],
        "action_key": "causal_expert_executed_actions",
    }
    source = tmp_path / "base.json"
    source.write_text(module.json.dumps(base), encoding="utf-8")
    correction = {"path": "correction.npz", "rows": 3, "sha256": "b" * 64}
    path = module.cumulative_manifest(source, [], [correction], tmp_path / "dataset")
    result = module.json.loads(path.read_text(encoding="utf-8"))
    assert result["transitions"] == 13
    assert result["chunks"] == [base["chunks"][0], correction]
    assert result["source"].startswith("causal demonstrations")


def test_correction_writer_rejects_overwrite(tmp_path: Path) -> None:
    module = load_script()
    writer = module.CorrectionWriter(tmp_path, chunk_size=1)
    observation = {
        "map": np.zeros((11, 16, 16), dtype=np.float32),
        "state": np.zeros(17, dtype=np.float32),
        "action_mask": np.ones(5, dtype=np.int8),
        "expert_action": 0,
        "episode_id": "episode",
    }
    writer.add(observation)
    second = module.CorrectionWriter(tmp_path, chunk_size=1)
    try:
        second.add(observation)
    except FileExistsError:
        pass
    else:
        raise AssertionError("Expected immutable DAgger chunk overwrite rejection")
