from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def derive(payload: dict[str, Any]) -> dict[str, Any]:
    events = payload.get("events")
    if not isinstance(events, list):
        return {
            "status": "UNAVAILABLE",
            "reason": "Source report does not contain step-resolved safety events.",
            "legacy_raw_safety_cost": payload.get(
                "legacy_raw_safety_cost", payload.get("mean_safety_cost")
            ),
            "legacy_diagnostic_label": "LEGACY_INACTIVITY_BIASED_DIAGNOSTIC",
        }
    required = {
        "event_type",
        "severity",
        "duration",
        "integrated_severity",
        "resolved",
    }
    if any(not required <= event.keys() for event in events):
        return {
            "status": "UNAVAILABLE",
            "reason": "Source events lack fields required by Safety Contract v3.",
            "legacy_raw_safety_cost": payload.get("legacy_raw_safety_cost"),
            "legacy_diagnostic_label": "LEGACY_INACTIVITY_BIASED_DIAGNOSTIC",
        }
    hard = [event for event in events if event["severity"] == "hard"]
    soft = [event for event in events if event["severity"] == "soft"]
    controlled = [event for event in events if event["severity"] == "controlled"]
    hard_cost = sum(float(event["integrated_severity"]) for event in hard)
    soft_cost = sum(float(event["integrated_severity"]) for event in soft)
    controlled_cost = sum(float(event["integrated_severity"]) for event in controlled)
    event_cost = hard_cost + soft_cost + controlled_cost
    success = payload.get("success")
    progress = payload.get("mission_progress")
    steps = payload.get("steps")
    return {
        "status": "DERIVED",
        "legacy_raw_safety_cost": payload.get("legacy_raw_safety_cost"),
        "hard_safety_cost_v3": hard_cost,
        "soft_safety_cost_v3": soft_cost,
        "event_safety_cost_v3": event_cost,
        "success_conditioned_safety_cost_v3": event_cost if success is True else None,
        "mission_progress_normalized_cost_v3": (
            event_cost / float(progress) if progress not in (None, 0) else None
        ),
        "cost_per_100_steps_v3": (
            100.0 * event_cost / int(steps) if steps not in (None, 0) else None
        ),
        "hard_event_count": len(hard),
        "soft_event_count": len(soft),
        "controlled_event_count": len(controlled),
        "unresolved_hard_event_count": sum(not event["resolved"] for event in hard),
        "legacy_diagnostic_label": "LEGACY_INACTIVITY_BIASED_DIAGNOSTIC",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = json.loads(args.input.read_text(encoding="utf-8"))
    result = derive(payload)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"SAFETY_CONTRACT_V3_MIGRATION={result['status']}")


if __name__ == "__main__":
    main()
