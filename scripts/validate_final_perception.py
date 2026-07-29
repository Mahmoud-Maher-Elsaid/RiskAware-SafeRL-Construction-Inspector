from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    config = json.loads(
        (ROOT / "configs/perception/stage5b_live_perception.json").read_text(encoding="utf-8")
    )
    model = ROOT / config["model"]["relative_path"]
    actual_hash = sha256(model)
    if actual_hash.lower() != config["model"]["sha256"].lower():
        raise RuntimeError("Perception checkpoint hash mismatch")
    runtime = json.loads(
        (ROOT / "reports/final_project_completion/final_runtime_summary.json").read_text(
            encoding="utf-8"
        )
    )
    required_true = (
        "cv_model_connected",
        "cuda_inference_verified",
        "perception_live_during_mission",
        "perception_affects_runtime_state",
    )
    if not all(runtime[key] is True for key in required_true):
        raise RuntimeError("Final perception runtime flags are not all true")
    if runtime["failure_count"] != 0 or runtime["inference_count"] < 1:
        raise RuntimeError("Final perception runtime counts are invalid")
    output = ROOT / "reports/final_submission/stage7_perception"
    output.mkdir(parents=True, exist_ok=True)
    annotation = (
        ROOT
        / "reports/final_project_completion/stages/stage-5c-rl-motor-control/evidence/annotated_000.png"
    )
    shutil.copy2(annotation, output / "selected_runtime_annotation.png")
    result = {
        "status": "PASSED",
        "model_sha256": actual_hash,
        "expected_classes": config["model"]["expected_classes"],
        "class_count": len(config["model"]["expected_classes"]),
        "device": config["runtime"]["device"],
        "cuda_inference_verified": runtime["cuda_inference_verified"],
        "inference_count": runtime["inference_count"],
        "failure_count": runtime["failure_count"],
        "annotated_frame_count": runtime["annotated_frame_count"],
        "perception_affects_runtime_state": runtime["perception_affects_runtime_state"],
        "perception_state_change_count": runtime["perception_state_change_count"],
        "unsupported_cv_classes": [
            "hole",
            "machine",
            "restricted_sign",
            "scaffolding_state",
        ],
        "unsupported_class_source": "simulator_ground_truth_only",
    }
    (output / "validation.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print("FINAL_PERCEPTION_VALIDATION=PASSED")


if __name__ == "__main__":
    main()
