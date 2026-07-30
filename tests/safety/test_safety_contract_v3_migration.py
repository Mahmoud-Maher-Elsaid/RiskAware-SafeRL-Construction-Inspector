from scripts.migrate_safety_contract_v3 import derive


def test_migration_marks_aggregate_only_report_unavailable() -> None:
    result = derive({"mean_safety_cost": 17.13})
    assert result["status"] == "UNAVAILABLE"
    assert result["legacy_raw_safety_cost"] == 17.13


def test_migration_derives_only_from_complete_events() -> None:
    result = derive(
        {
            "legacy_raw_safety_cost": 3.0,
            "success": True,
            "mission_progress": 1.0,
            "steps": 100,
            "events": [
                {
                    "event_type": "restricted",
                    "severity": "hard",
                    "duration": 1,
                    "integrated_severity": 1.0,
                    "resolved": True,
                },
                {
                    "event_type": "near",
                    "severity": "soft",
                    "duration": 2,
                    "integrated_severity": 0.5,
                    "resolved": True,
                },
            ],
        }
    )
    assert result["status"] == "DERIVED"
    assert result["hard_safety_cost_v3"] == 1.0
    assert result["event_safety_cost_v3"] == 1.5
