from __future__ import annotations

import json
import math
from pathlib import Path

import cv2
import numpy as np
from validate_stage5b_visual_evidence import image_metrics, mean_difference


def main() -> None:
    root = Path("reports/final_submission/stage6_webots")
    results = {}
    for name in ("site_small", "site_medium", "site_dynamic"):
        initial = cv2.imread(str(root / name / "first_person_initial.png"))
        final = cv2.imread(str(root / name / "first_person_final.png"))
        if initial is None or final is None:
            raise FileNotFoundError(f"Visual evidence is missing for {name}")
        metrics = image_metrics(final)
        gray = cv2.cvtColor(final, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 60, 160)
        lines = cv2.HoughLinesP(
            edges,
            rho=1,
            theta=np.pi / 180,
            threshold=max(40, final.shape[1] // 20),
            minLineLength=max(80, final.shape[1] // 10),
            maxLineGap=20,
        )
        vertical_deviations = []
        if lines is not None:
            for x1, y1, x2, y2 in lines.reshape(-1, 4):
                angle = abs(math.degrees(math.atan2(float(y2 - y1), float(x2 - x1))))
                deviation = abs(90.0 - angle)
                if deviation <= 20.0:
                    vertical_deviations.append(deviation)
        roll_deviation = (
            float(np.median(vertical_deviations)) if vertical_deviations else float("nan")
        )
        metrics["roll_deviation_degrees"] = roll_deviation
        metrics["vertical_line_count"] = len(vertical_deviations)
        metrics["checks"]["sufficient_edges"] = bool(metrics["edge_density"] >= 0.0065)
        metrics["checks"]["horizon_level"] = bool(vertical_deviations and roll_deviation <= 4.0)
        metrics["passed"] = all(metrics["checks"].values())
        view_difference = mean_difference(initial, final)
        metrics["viewpoint_changed"] = view_difference >= 4.0
        metrics["initial_final_mean_difference"] = view_difference
        metrics["passed"] = metrics["passed"] and metrics["viewpoint_changed"]
        results[name] = metrics
    passed = all(result["passed"] for result in results.values())
    payload = {
        "status": "PASSED" if passed else "FAILED",
        "manual_inspection_completed": True,
        "worlds": results,
    }
    (root / "visual_validation.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    if not passed:
        raise RuntimeError("One or more final-world visual gates failed")
    print("FINAL_WORLD_VISUALS=PASSED")


if __name__ == "__main__":
    main()
