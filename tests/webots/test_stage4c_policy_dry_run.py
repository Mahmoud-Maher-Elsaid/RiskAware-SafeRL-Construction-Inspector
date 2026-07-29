from __future__ import annotations

import json

import numpy as np
import pytest

from riskaware_saferrl.webots.policy_dry_run import (
    PolicyDryRunEngine,
)


class FakePolicy:
    pass


class FakeModel:
    def __init__(
        self,
        action: int,
    ) -> None:
        self.action = action
        self.policy = FakePolicy()
        self.device = "cpu"
        self.last_deterministic: bool | None = None
        self.last_mask: np.ndarray | None = None

    def predict(
        self,
        observation: dict[str, np.ndarray],
        *,
        deterministic: bool,
        action_masks: np.ndarray,
    ) -> tuple[np.ndarray, None]:
        assert set(observation) == {
            "map",
            "state",
        }

        self.last_deterministic = deterministic
        self.last_mask = action_masks.copy()

        return (
            np.asarray(
                self.action,
                dtype=np.int64,
            ),
            None,
        )


def valid_observation() -> dict[str, np.ndarray]:
    semantic_map = np.zeros(
        7 * 12 * 12,
        dtype=np.float32,
    )

    semantic_map[5] = 1.0

    state = np.asarray(
        [
            0.5,
            0.5,
            0.1,
            0.0,
        ],
        dtype=np.float32,
    )

    return {
        "map": semantic_map,
        "state": state,
    }


def test_policy_engine_creates_valid_dry_run_proposal() -> None:
    model = FakeModel(action=3)

    engine = PolicyDryRunEngine(model)

    proposal = engine.propose(
        sample_index=7,
        observation=valid_observation(),
        action_mask=np.asarray(
            [
                True,
                False,
                True,
                True,
                False,
            ],
            dtype=np.bool_,
        ),
        deterministic=True,
    )

    assert proposal.sample_index == 7
    assert proposal.action == 3
    assert proposal.action_name == "MOVE_RIGHT"
    assert proposal.valid_action_count == 3
    assert proposal.mask_respected is True
    assert proposal.motors_connected is False
    assert model.last_deterministic is True
    assert model.last_mask is not None


def test_policy_engine_rejects_masked_policy_action() -> None:
    engine = PolicyDryRunEngine(FakeModel(action=1))

    with pytest.raises(
        RuntimeError,
        match="masked-out",
    ):
        engine.propose(
            sample_index=0,
            observation=valid_observation(),
            action_mask=np.asarray(
                [
                    True,
                    False,
                    True,
                    False,
                    False,
                ],
                dtype=np.bool_,
            ),
        )


def test_policy_engine_rejects_invalid_observation_shape() -> None:
    engine = PolicyDryRunEngine(FakeModel(action=0))

    observation = valid_observation()
    observation["map"] = np.zeros(
        10,
        dtype=np.float32,
    )

    with pytest.raises(
        ValueError,
        match="1008",
    ):
        engine.propose(
            sample_index=0,
            observation=observation,
            action_mask=np.ones(
                5,
                dtype=np.bool_,
            ),
        )


def test_policy_engine_rejects_empty_action_mask() -> None:
    engine = PolicyDryRunEngine(FakeModel(action=0))

    with pytest.raises(
        ValueError,
        match="valid action",
    ):
        engine.propose(
            sample_index=0,
            observation=valid_observation(),
            action_mask=np.zeros(
                5,
                dtype=np.bool_,
            ),
        )


def test_policy_proposal_serialization_preserves_safety_boundary() -> None:
    engine = PolicyDryRunEngine(FakeModel(action=4))

    proposal = engine.propose(
        sample_index=3,
        observation=valid_observation(),
        action_mask=np.asarray(
            [
                False,
                False,
                False,
                False,
                True,
            ],
            dtype=np.bool_,
        ),
        deterministic=False,
    )

    payload = json.loads(proposal.to_json())

    assert payload["action"] == 4
    assert payload["action_name"] == "INSPECT"
    assert payload["mask_respected"] is True
    assert payload["motors_connected"] is False
