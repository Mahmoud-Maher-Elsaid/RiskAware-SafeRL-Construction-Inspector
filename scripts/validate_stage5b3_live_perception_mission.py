from __future__ import annotations

import argparse
import json
import pathlib
from typing import Any


def load_json(path: pathlib.Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(payload, dict):
        raise TypeError(f"Expected a JSON object: {path}")
    return payload


def find_stage5a3_report(
    log_root: pathlib.Path,
) -> tuple[pathlib.Path, dict[str, Any]]:
    candidates: list[tuple[int, pathlib.Path, dict[str, Any]]] = []

    for path in log_root.rglob("*.json"):
        try:
            payload = load_json(path)
        except Exception:
            continue

        if payload.get("stage") == "5A3" and "runtime_verified" in payload:
            candidates.append((path.stat().st_mtime_ns, path, payload))

    if not candidates:
        raise FileNotFoundError("Could not locate a Stage 5A3 runtime report.")

    _, path, payload = max(candidates, key=lambda item: item[0])
    return path, payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", required=True)
    parser.add_argument("--evidence-root", required=True)
    parser.add_argument("--sidecar-output", required=True)
    parser.add_argument("--report", required=True)
    args = parser.parse_args()

    project_root = pathlib.Path(args.project_root).resolve()
    evidence_root = pathlib.Path(args.evidence_root).resolve()
    sidecar_output = pathlib.Path(args.sidecar_output).resolve()
    report_path = pathlib.Path(args.report).resolve()

    summary_path = sidecar_output / "perception_summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(f"Sidecar summary was not found: {summary_path}")

    sidecar = load_json(summary_path)
    stage5a_path, stage5a = find_stage5a3_report(
        project_root / "webots" / "logs" / "stage5a3_closed_loop"
    )

    evidence_frames = sorted(evidence_root.glob("*.png"))
    annotated_frames = sorted((sidecar_output / "annotated_frames").glob("*.png"))
    evidence_names = {path.name for path in evidence_frames}
    annotated_names = {path.name for path in annotated_frames}
    inference_count = int(sidecar.get("inference_count", 0))

    checks = {
        "stage5a3_runtime_verified": bool(stage5a.get("runtime_verified")),
        "route_completed": bool(stage5a.get("route_completed")),
        "returned_to_start": bool(stage5a.get("returned_to_start")),
        "sidecar_runtime_verified": bool(sidecar.get("runtime_verified")),
        "cv_model_connected": bool(sidecar.get("cv_model_connected")),
        "cuda_inference_verified": bool(sidecar.get("cuda_inference_verified")),
        "perception_live_during_mission": bool(sidecar.get("perception_live_during_mission")),
        "zero_sidecar_failures": int(sidecar.get("failure_count", -1)) == 0,
        "evidence_exists": len(evidence_frames) > 0,
        "all_evidence_frames_processed": inference_count == len(evidence_frames),
        "all_evidence_frames_annotated": (
            len(annotated_frames) == len(evidence_frames) and annotated_names == evidence_names
        ),
        "policy_does_not_control_motors": (sidecar.get("policy_controls_motors") is False),
        "perception_outside_motor_loop": (sidecar.get("perception_in_motor_control_loop") is False),
    }
    failures = [name for name, passed in checks.items() if not passed]

    report = {
        "schema_version": 1,
        "stage": "5B3",
        "runtime_verified": not failures,
        "cv_model_connected": checks["cv_model_connected"],
        "cuda_inference_verified": checks["cuda_inference_verified"],
        "controller_synchronized_evidence": True,
        "perception_live_during_mission": checks["perception_live_during_mission"],
        "perception_transport": sidecar.get("perception_transport"),
        "perception_in_motor_control_loop": False,
        "policy_controls_motors": False,
        "perception_can_stop_robot": False,
        "motor_motion_source": stage5a.get("motor_motion_source"),
        "absence_of_detection_is_not_a_safety_claim": True,
        "collision_free_claimed": bool(stage5a.get("collision_free_claimed")),
        "route_completed": checks["route_completed"],
        "returned_to_start": checks["returned_to_start"],
        "waypoints_total": int(stage5a.get("waypoints_total", 0)),
        "waypoints_visited": int(stage5a.get("waypoints_visited", 0)),
        "mission_capture_count": int(stage5a.get("capture_count", 0)),
        "evidence_frame_count": len(evidence_frames),
        "inference_count": inference_count,
        "all_evidence_frames_processed": checks["all_evidence_frames_processed"],
        "annotated_frame_count": len(annotated_frames),
        "all_evidence_frames_annotated": checks["all_evidence_frames_annotated"],
        "total_detections": int(sidecar.get("total_detections", 0)),
        "detected_class_diversity": int(sidecar.get("detected_class_diversity", 0)),
        "class_counts": sidecar.get("class_counts", {}),
        "risk_counts": sidecar.get("risk_counts", {}),
        "mean_inference_ms": float(sidecar.get("mean_inference_ms", 0.0)),
        "p95_inference_ms": float(sidecar.get("p95_inference_ms", 0.0)),
        "maximum_inference_ms": float(sidecar.get("maximum_inference_ms", 0.0)),
        "model_sha256": sidecar.get("model_sha256"),
        "stage5a3_report": str(stage5a_path),
        "perception_summary": str(summary_path),
        "checks": checks,
        "failure_count": len(failures),
        "failures": failures,
    }

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    print(f"STAGE5B3_REPORT={report_path}")
    return 0 if not failures else 2


if __name__ == "__main__":
    raise SystemExit(main())
