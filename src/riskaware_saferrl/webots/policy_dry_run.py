from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from riskaware_saferrl.webots.bridge import GridAction

POLICY_PROPOSAL_SCHEMA_VERSION = 1
EXPECTED_MAP_LENGTH = 7 * 12 * 12
EXPECTED_STATE_LENGTH = 4
EXPECTED_ACTION_COUNT = 5


class PredictivePolicy(Protocol):
    def predict(
        self,
        observation: dict[str, np.ndarray],
        *,
        deterministic: bool,
        action_masks: np.ndarray,
    ) -> tuple[Any, Any]: ...


@dataclass(frozen=True)
class PolicyProposal:
    schema_version: int
    sample_index: int
    action: int
    action_name: str
    deterministic: bool
    action_mask: tuple[bool, bool, bool, bool, bool]
    valid_action_count: int
    mask_respected: bool
    motors_connected: bool = False

    def __post_init__(self) -> None:
        if self.schema_version != POLICY_PROPOSAL_SCHEMA_VERSION:
            raise ValueError("Unsupported policy proposal schema version.")

        if self.sample_index < 0:
            raise ValueError("sample_index must be non-negative.")

        if not 0 <= self.action < EXPECTED_ACTION_COUNT:
            raise ValueError("The proposed action is outside the valid range.")

        expected_action_name = GridAction(self.action).name

        if self.action_name != expected_action_name:
            raise ValueError("The action name does not match the action index.")

        if len(self.action_mask) != EXPECTED_ACTION_COUNT:
            raise ValueError("The action mask must contain five values.")

        if not any(self.action_mask):
            raise ValueError("The action mask must contain a valid action.")

        if not self.action_mask[self.action]:
            raise ValueError("The proposed action is masked out.")

        expected_valid_action_count = sum(self.action_mask)

        if self.valid_action_count != expected_valid_action_count:
            raise ValueError("valid_action_count does not match the mask.")

        if not self.mask_respected:
            raise ValueError("Policy proposals must respect the action mask.")

        if self.motors_connected:
            raise ValueError("Dry-run policy proposals cannot control motors.")

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "sample_index": self.sample_index,
            "action": self.action,
            "action_name": self.action_name,
            "deterministic": self.deterministic,
            "action_mask": list(self.action_mask),
            "valid_action_count": self.valid_action_count,
            "mask_respected": self.mask_respected,
            "motors_connected": self.motors_connected,
        }

    def to_json(self) -> str:
        return json.dumps(
            self.to_dict(),
            separators=(",", ":"),
            sort_keys=True,
        )


class PolicyDryRunEngine:
    def __init__(
        self,
        model: PredictivePolicy,
        *,
        checkpoint_path: Path | None = None,
        checkpoint_sha256: str | None = None,
    ) -> None:
        self._model = model
        self._checkpoint_path = checkpoint_path
        self._checkpoint_sha256 = checkpoint_sha256

    @staticmethod
    def checkpoint_sha256(
        path: Path,
    ) -> str:
        digest = hashlib.sha256()

        with path.open("rb") as checkpoint_file:
            for chunk in iter(
                lambda: checkpoint_file.read(1024 * 1024),
                b"",
            ):
                digest.update(chunk)

        return digest.hexdigest().upper()

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str | Path,
        *,
        expected_sha256: str | None = None,
        device: str = "cpu",
        seed: int = 42,
    ) -> PolicyDryRunEngine:
        from sb3_contrib import MaskablePPO

        resolved_path = Path(checkpoint_path).resolve()

        if not resolved_path.is_file():
            raise FileNotFoundError(f"Checkpoint was not found: {resolved_path}")

        actual_sha256 = cls.checkpoint_sha256(resolved_path)

        if expected_sha256 is not None and actual_sha256 != expected_sha256.upper():
            raise ValueError(
                "Checkpoint SHA256 mismatch. "
                f"Expected {expected_sha256.upper()}, "
                f"received {actual_sha256}."
            )

        model = MaskablePPO.load(
            str(resolved_path),
            device=device,
            print_system_info=False,
        )

        model.policy.set_training_mode(False)
        model.set_random_seed(seed)

        cls._validate_checkpoint_spaces(model)

        return cls(
            model,
            checkpoint_path=resolved_path,
            checkpoint_sha256=actual_sha256,
        )

    @staticmethod
    def _validate_checkpoint_spaces(
        model: object,
    ) -> None:
        action_space = getattr(
            model,
            "action_space",
            None,
        )

        if action_space is None or not hasattr(action_space, "n"):
            raise ValueError("The checkpoint action space must be Discrete.")

        if int(action_space.n) != EXPECTED_ACTION_COUNT:
            raise ValueError("The checkpoint must contain five actions.")

        observation_space = getattr(
            model,
            "observation_space",
            None,
        )

        if observation_space is None or not hasattr(
            observation_space,
            "spaces",
        ):
            raise ValueError("The checkpoint observation space must be Dict.")

        observation_spaces = observation_space.spaces

        if set(observation_spaces) != {
            "map",
            "state",
        }:
            raise ValueError("The checkpoint observation keys must be map and state.")

        map_shape = tuple(int(value) for value in observation_spaces["map"].shape)

        state_shape = tuple(int(value) for value in observation_spaces["state"].shape)

        if map_shape != (EXPECTED_MAP_LENGTH,):
            raise ValueError("The checkpoint map shape must be (1008,).")

        if state_shape != (EXPECTED_STATE_LENGTH,):
            raise ValueError("The checkpoint state shape must be (4,).")

        if np.dtype(observation_spaces["map"].dtype) != np.dtype(np.float32):
            raise ValueError("The checkpoint map dtype must be float32.")

        if np.dtype(observation_spaces["state"].dtype) != np.dtype(np.float32):
            raise ValueError("The checkpoint state dtype must be float32.")

    @property
    def algorithm_class(self) -> str:
        return self._model.__class__.__name__

    @property
    def policy_class(self) -> str:
        policy = getattr(
            self._model,
            "policy",
            None,
        )

        if policy is None:
            return "UnknownPolicy"

        return policy.__class__.__name__

    @property
    def device(self) -> str:
        return str(
            getattr(
                self._model,
                "device",
                "unknown",
            )
        )

    @property
    def checkpoint_path(self) -> str | None:
        if self._checkpoint_path is None:
            return None

        return str(self._checkpoint_path)

    @property
    def checkpoint_hash(self) -> str | None:
        return self._checkpoint_sha256

    @staticmethod
    def _validated_observation(
        observation: dict[str, np.ndarray],
    ) -> dict[str, np.ndarray]:
        if set(observation) != {
            "map",
            "state",
        }:
            raise ValueError("Observation must contain map and state.")

        map_values = np.asarray(
            observation["map"],
            dtype=np.float32,
        ).reshape(-1)

        state_values = np.asarray(
            observation["state"],
            dtype=np.float32,
        ).reshape(-1)

        if map_values.size != EXPECTED_MAP_LENGTH:
            raise ValueError("The semantic map must contain 1008 values.")

        if state_values.size != EXPECTED_STATE_LENGTH:
            raise ValueError("The state vector must contain four values.")

        if not all(math.isfinite(float(value)) for value in map_values):
            raise ValueError("The semantic map contains non-finite values.")

        if not all(math.isfinite(float(value)) for value in state_values):
            raise ValueError("The state vector contains non-finite values.")

        if not bool(np.all((map_values >= 0.0) & (map_values <= 1.0))):
            raise ValueError("The semantic map must be normalized.")

        if not bool(np.all((state_values >= 0.0) & (state_values <= 1.0))):
            raise ValueError("The state vector must be normalized.")

        return {
            "map": map_values,
            "state": state_values,
        }

    @staticmethod
    def _validated_mask(
        action_mask: np.ndarray,
    ) -> np.ndarray:
        mask_values = np.asarray(
            action_mask,
            dtype=np.bool_,
        ).reshape(-1)

        if mask_values.size != EXPECTED_ACTION_COUNT:
            raise ValueError("The action mask must contain five values.")

        if not bool(np.any(mask_values)):
            raise ValueError("The action mask must contain a valid action.")

        return mask_values

    def propose(
        self,
        *,
        sample_index: int,
        observation: dict[str, np.ndarray],
        action_mask: np.ndarray,
        deterministic: bool = True,
    ) -> PolicyProposal:
        if sample_index < 0:
            raise ValueError("sample_index must be non-negative.")

        validated_observation = self._validated_observation(observation)

        validated_mask = self._validated_mask(action_mask)

        action, _ = self._model.predict(
            validated_observation,
            deterministic=deterministic,
            action_masks=validated_mask,
        )

        action_values = np.asarray(action).reshape(-1)

        if action_values.size != 1:
            raise RuntimeError("The policy returned a non-scalar action.")

        action_index = int(action_values[0])

        if not 0 <= action_index < EXPECTED_ACTION_COUNT:
            raise RuntimeError("The policy returned an invalid action index.")

        mask_respected = bool(validated_mask[action_index])

        if not mask_respected:
            raise RuntimeError("The policy selected a masked-out action.")

        mask_tuple = tuple(bool(value) for value in validated_mask)

        return PolicyProposal(
            schema_version=(POLICY_PROPOSAL_SCHEMA_VERSION),
            sample_index=sample_index,
            action=action_index,
            action_name=GridAction(action_index).name,
            deterministic=bool(deterministic),
            action_mask=mask_tuple,
            valid_action_count=int(np.count_nonzero(validated_mask)),
            mask_respected=mask_respected,
            motors_connected=False,
        )
