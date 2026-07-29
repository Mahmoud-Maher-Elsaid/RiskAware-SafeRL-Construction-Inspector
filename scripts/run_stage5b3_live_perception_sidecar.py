from __future__ import annotations

import argparse
import json
import pathlib
import time
import traceback

from riskaware_saferrl.live_perception import create_live_perception_backend
from riskaware_saferrl.live_perception_sidecar import (
    LivePerceptionSidecar,
    write_json_atomic,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--evidence-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--stop-file", required=True)
    parser.add_argument("--timeout-seconds", type=float, default=900.0)
    args = parser.parse_args()

    project_root = pathlib.Path(args.project_root).resolve()
    config_path = pathlib.Path(args.config).resolve()
    evidence_root = pathlib.Path(args.evidence_root).resolve()
    output_root = pathlib.Path(args.output).resolve()
    stop_file = pathlib.Path(args.stop_file).resolve()

    output_root.mkdir(parents=True, exist_ok=True)
    ready_path = output_root / "perception_ready.json"
    failure_path = output_root / "perception_failure.json"

    for path in (ready_path, failure_path):
        if path.exists():
            path.unlink()

    try:
        not_before_ns = time.time_ns()
        backend = create_live_perception_backend(
            project_root=project_root,
            config_path=config_path,
        )
        warmup_ms = backend.warmup()

        ready_payload = {
            "schema_version": 1,
            "stage": "5B3",
            "ready": True,
            "cv_model_connected": True,
            "backend": backend.backend_name,
            "device": backend.device,
            "cuda_inference_verified": str(backend.device).startswith("cuda"),
            "model_path": str(backend.model_path),
            "model_sha256": backend.model_sha256,
            "warmup_ms": warmup_ms,
            "not_before_ns": not_before_ns,
        }
        write_json_atomic(ready_path, ready_payload)

        print("STAGE5B3_PERCEPTION_READY", flush=True)
        print(json.dumps(ready_payload, indent=2, sort_keys=True), flush=True)

        sidecar = LivePerceptionSidecar(
            backend=backend,
            evidence_root=evidence_root,
            output_root=output_root,
            not_before_ns=not_before_ns,
        )
        summary = sidecar.run(
            stop_file=stop_file,
            warmup_ms=warmup_ms,
            timeout_seconds=args.timeout_seconds,
        )

        print("STAGE5B3_PERCEPTION_COMPLETE", flush=True)
        print(json.dumps(summary, indent=2, sort_keys=True), flush=True)

        if not summary["runtime_verified"]:
            return 2
        if not summary["cv_model_connected"]:
            return 3
        if not summary["cuda_inference_verified"]:
            return 4
        if not summary["perception_live_during_mission"]:
            return 5
        return 0
    except Exception as exc:
        failure_payload = {
            "schema_version": 1,
            "stage": "5B3",
            "ready": False,
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(),
        }
        write_json_atomic(failure_path, failure_payload)
        print(json.dumps(failure_payload, indent=2, sort_keys=True), flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
