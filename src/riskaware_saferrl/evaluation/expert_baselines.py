from __future__ import annotations

import hashlib
import random
from collections.abc import Sequence
from typing import Any

from riskaware_saferrl.envs import ConstructionInspectionEnv
from riskaware_saferrl.evaluation.reward_audit import (
    REWARD_COMPONENT_NAMES,
    RewardAuditWrapper,
)
from riskaware_saferrl.planners import (
    InspectionPlan,
    build_oracle_inspection_plan,
)
from riskaware_saferrl.scenarios import Scenario


def scenario_seed(scenario_id: str, base_seed: int) -> int:
    payload = f"{scenario_id}:{base_seed}".encode()
    digest = hashlib.sha256(payload).digest()
    return int.from_bytes(digest[:8], byteorder="big")


def run_action_sequence(
    scenario: Scenario,
    actions: Sequence[int],
) -> dict[str, Any]:
    environment = RewardAuditWrapper(ConstructionInspectionEnv(scenario=scenario))

    total_reward = 0.0
    total_cost = 0.0
    collision_cost = 0.0
    worker_cost = 0.0
    restricted_cost = 0.0
    reward_components = {component_name: 0.0 for component_name in REWARD_COMPONENT_NAMES}
    executed_actions = 0

    _, initial_info = environment.reset(seed=0)
    final_info = initial_info

    for action in actions:
        _, reward, terminated, truncated, info = environment.step(int(action))

        final_info = info
        total_reward += float(reward)
        total_cost += float(info["cost"])
        collision_cost += float(info["cost_collision"])
        worker_cost += float(info["cost_worker"])
        restricted_cost += float(info["cost_restricted"])
        executed_actions += 1

        for component_name, value in info["reward_components"].items():
            reward_components[component_name] += float(value)

        if terminated or truncated:
            break

    environment.close()

    return {
        "scenario_id": scenario.scenario_id,
        "split": scenario.split,
        "reward": total_reward,
        "safety_cost": total_cost,
        "collision_cost": collision_cost,
        "worker_cost": worker_cost,
        "restricted_cost": restricted_cost,
        "hazard_recall": float(final_info["hazard_recall"]),
        "coverage": float(final_info["coverage"]),
        "success": float(bool(final_info["success"])),
        "steps": float(executed_actions),
        **reward_components,
    }


def evaluate_plan(
    scenario: Scenario,
    plan: InspectionPlan,
) -> dict[str, Any]:
    record = run_action_sequence(
        scenario,
        plan.actions,
    )

    record.update(
        {
            "planner": plan.planner_name,
            "plan_complete": float(plan.complete),
            "planned_actions": float(len(plan.actions)),
            "planned_hazards": float(len(plan.visited_hazards)),
            "unreachable_hazards": float(len(plan.unreachable_hazards)),
        }
    )

    return record


def evaluate_inspect_only(
    scenario: Scenario,
) -> dict[str, Any]:
    record = run_action_sequence(
        scenario,
        [4] * scenario.max_steps,
    )
    record["planner"] = "inspect_only"
    record["plan_complete"] = 0.0
    return record


def evaluate_random(
    scenario: Scenario,
    *,
    base_seed: int,
) -> dict[str, Any]:
    generator = random.Random(
        scenario_seed(
            scenario.scenario_id,
            base_seed,
        )
    )

    actions = [generator.randrange(5) for _ in range(scenario.max_steps)]

    record = run_action_sequence(
        scenario,
        actions,
    )
    record["planner"] = "random"
    record["plan_complete"] = 0.0
    return record


def evaluate_all_baselines(
    scenarios: Sequence[Scenario],
    *,
    random_seed: int,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []

    for scenario in scenarios:
        records.append(evaluate_inspect_only(scenario))
        records.append(
            evaluate_random(
                scenario,
                base_seed=random_seed,
            )
        )

        for safety_aware in (False, True):
            plan = build_oracle_inspection_plan(
                scenario,
                safety_aware=safety_aware,
            )
            records.append(
                evaluate_plan(
                    scenario,
                    plan,
                )
            )

    return records
