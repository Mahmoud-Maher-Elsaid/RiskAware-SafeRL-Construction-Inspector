from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from typing import Protocol

import numpy as np

from riskaware_saferrl.webots.bridge import ActionBridge, CardinalHeading
from riskaware_saferrl.webots.motion_primitives import DifferentialDriveMapper
from riskaware_saferrl.webots.policy_dry_run import PolicyDryRunEngine


class ShieldEvaluator(Protocol):
    def evaluate(self, proposed_action: int) -> tuple[int, str, float]: ...


@dataclass(frozen=True)
class SafeExecutionTrace:
    sample_index: int
    observation_map_sum: float
    observation_state: tuple[float, ...]
    valid_action_mask: tuple[bool, ...]
    proposed_action: int
    shield_decision: str
    fallback_decision: str | None
    executed_action: int
    safety_cost: float
    motion_primitives: tuple[str, ...]
    motor_commands: tuple[tuple[float, float], ...]
    policy_controls_motors: bool
    inference_latency_ms: float
    step_duration_ms: float

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


class SafePolicyPipeline:
    """Execute the mandatory masked-policy, shield, fallback, action order."""

    def __init__(
        self,
        policy: PolicyDryRunEngine,
        shield: ShieldEvaluator,
        *,
        policy_controls_motors: bool = False,
        verified_motor_runtime: bool = False,
    ) -> None:
        if policy_controls_motors and not verified_motor_runtime:
            raise ValueError("Policy motor control requires the dedicated verified runtime gate.")
        self.policy = policy
        self.shield = shield
        self.policy_controls_motors = policy_controls_motors

    def execute(
        self,
        *,
        sample_index: int,
        observation: dict[str, np.ndarray],
        task_valid_mask: np.ndarray,
        heading: CardinalHeading,
    ) -> SafeExecutionTrace:
        started = time.perf_counter()

        # The ordering here is intentionally explicit and covered by integration tests.
        proposal_started = time.perf_counter()
        proposal = self.policy.propose(
            sample_index=sample_index,
            observation=observation,
            action_mask=task_valid_mask,
            deterministic=True,
        )
        inference_latency = (time.perf_counter() - proposal_started) * 1000.0

        executed_action, resolution, safety_cost = self.shield.evaluate(proposal.action)
        fallback = resolution if resolution in {"emergency_hold", "least_unsafe"} else None

        primitives = ActionBridge().plan(executed_action, heading)
        mapper = DifferentialDriveMapper(forward_velocity=1.0, turn_velocity=0.3)
        motor_commands = tuple(
            (
                mapper.command_for(primitive).left_velocity,
                mapper.command_for(primitive).right_velocity,
            )
            for primitive in primitives
        )
        duration = (time.perf_counter() - started) * 1000.0
        return SafeExecutionTrace(
            sample_index=sample_index,
            observation_map_sum=float(np.asarray(observation["map"]).sum()),
            observation_state=tuple(float(value) for value in observation["state"]),
            valid_action_mask=tuple(bool(value) for value in task_valid_mask),
            proposed_action=proposal.action,
            shield_decision=resolution,
            fallback_decision=fallback,
            executed_action=int(executed_action),
            safety_cost=float(safety_cost),
            motion_primitives=tuple(primitive.name for primitive in primitives),
            motor_commands=motor_commands,
            policy_controls_motors=self.policy_controls_motors,
            inference_latency_ms=inference_latency,
            step_duration_ms=duration,
        )
