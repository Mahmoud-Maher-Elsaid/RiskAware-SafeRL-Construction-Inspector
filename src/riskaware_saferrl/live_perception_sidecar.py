from __future__ import annotations

import json
import pathlib
import statistics
import time
from collections import Counter
from dataclasses import dataclass
from typing import Any

import cv2

from riskaware_saferrl.live_perception import (
    LivePerceptionResult,
    annotate_frame,
)


def percentile(values: list[float], quantile: float) -> float:
    if not values:
        return 0.0

    if not 0.0 <= quantile <= 1.0:
        raise ValueError("quantile must be in the interval [0, 1].")

    ordered = sorted(values)

    if len(ordered) == 1:
        return ordered[0]

    position = (len(ordered) - 1) * quantile
    lower_index = int(position)
    upper_index = min(lower_index + 1, len(ordered) - 1)
    fraction = position - lower_index

    return ordered[lower_index] * (1.0 - fraction) + ordered[upper_index] * fraction


def write_json_atomic(
    path: pathlib.Path,
    payload: dict[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    temporary_path = path.with_suffix(path.suffix + ".tmp")

    temporary_path.write_text(
        json.dumps(
            payload,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    temporary_path.replace(path)


@dataclass(frozen=True)
class ProcessedFrame:
    source_name: str
    annotated_name: str
    result: LivePerceptionResult

    def to_dict(self) -> dict[str, Any]:
        payload = self.result.to_dict()
        payload["source_name"] = self.source_name
        payload["annotated_name"] = self.annotated_name
        return payload


class LivePerceptionSidecar:
    def __init__(
        self,
        *,
        backend: Any,
        evidence_root: pathlib.Path,
        output_root: pathlib.Path,
        not_before_ns: int,
    ) -> None:
        self.backend = backend
        self.evidence_root = pathlib.Path(evidence_root).resolve()
        self.output_root = pathlib.Path(output_root).resolve()

        self.annotated_root = self.output_root / "annotated_frames"
        self.records_path = self.output_root / "perception_records.jsonl"
        self.summary_path = self.output_root / "perception_summary.json"

        self.not_before_ns = int(not_before_ns)

        self.processed_names: set[str] = set()
        self.candidate_sizes: dict[str, int] = {}
        self.processed_frames: list[ProcessedFrame] = []
        self.inference_times_ms: list[float] = []

        self.class_counter: Counter[str] = Counter()
        self.risk_counter: Counter[str] = Counter()

        self.failure_count = 0
        self.live_during_mission = False

        self.output_root.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.annotated_root.mkdir(
            parents=True,
            exist_ok=True,
        )

        if self.records_path.exists():
            self.records_path.unlink()

        if self.summary_path.exists():
            self.summary_path.unlink()

    def discover_candidates(
        self,
    ) -> list[pathlib.Path]:
        if not self.evidence_root.exists():
            return []

        candidates: list[pathlib.Path] = []

        minimum_mtime_ns = self.not_before_ns - 2_000_000_000

        for path in sorted(self.evidence_root.glob("*.png")):
            if path.name in self.processed_names:
                continue

            try:
                stat = path.stat()
            except OSError:
                continue

            if stat.st_mtime_ns < minimum_mtime_ns:
                continue

            if stat.st_size <= 0:
                continue

            previous_size = self.candidate_sizes.get(path.name)

            self.candidate_sizes[path.name] = stat.st_size

            if previous_size is None:
                continue

            if previous_size != stat.st_size:
                continue

            candidates.append(path)

        return candidates

    def process_frame(
        self,
        path: pathlib.Path,
        *,
        mission_active: bool,
    ) -> ProcessedFrame:
        frame = cv2.imread(
            str(path),
            cv2.IMREAD_COLOR,
        )

        if frame is None:
            raise RuntimeError(f"Could not decode controller evidence frame: {path}")

        frame_id = len(self.processed_frames)

        result = self.backend.infer(
            frame,
            frame_id=frame_id,
        )

        if not result.model_connected:
            raise RuntimeError("The Stage 5B backend reported model disconnection.")

        annotated = annotate_frame(
            frame,
            result,
        )

        annotated_path = self.annotated_root / path.name

        if not cv2.imwrite(
            str(annotated_path),
            annotated,
        ):
            raise RuntimeError(f"Could not save annotated frame: {annotated_path}")

        processed = ProcessedFrame(
            source_name=path.name,
            annotated_name=annotated_path.name,
            result=result,
        )

        with self.records_path.open(
            "a",
            encoding="utf-8",
            newline="\n",
        ) as handle:
            handle.write(
                json.dumps(
                    processed.to_dict(),
                    sort_keys=True,
                )
                + "\n"
            )

        self.processed_names.add(path.name)
        self.candidate_sizes.pop(
            path.name,
            None,
        )

        self.processed_frames.append(processed)
        self.inference_times_ms.append(result.inference_ms)

        self.risk_counter[result.risk.risk_level] += 1

        for detection in result.detections:
            self.class_counter[detection.normalized_class_name] += 1

        if mission_active:
            self.live_during_mission = True

        return processed

    def process_available_frames(
        self,
        *,
        mission_active: bool,
    ) -> int:
        processed_count = 0

        for path in self.discover_candidates():
            try:
                self.process_frame(
                    path,
                    mission_active=mission_active,
                )
            except Exception:
                self.failure_count += 1
                raise

            processed_count += 1

        return processed_count

    def build_summary(
        self,
        *,
        warmup_ms: float,
        stop_signal_observed: bool,
    ) -> dict[str, Any]:
        inference_count = len(self.processed_frames)

        mean_inference_ms = (
            statistics.fmean(self.inference_times_ms) if self.inference_times_ms else 0.0
        )

        maximum_inference_ms = max(self.inference_times_ms) if self.inference_times_ms else 0.0

        p95_inference_ms = percentile(
            self.inference_times_ms,
            0.95,
        )

        annotated_count = len(list(self.annotated_root.glob("*.png")))

        runtime_verified = (
            inference_count > 0 and self.failure_count == 0 and annotated_count == inference_count
        )

        return {
            "schema_version": 1,
            "stage": "5B3",
            "runtime_verified": runtime_verified,
            "cv_model_connected": bool(self.backend.model_connected),
            "cuda_inference_verified": str(self.backend.device).startswith("cuda"),
            "backend": self.backend.backend_name,
            "device": self.backend.device,
            "model_path": str(self.backend.model_path),
            "model_sha256": (self.backend.model_sha256),
            "frame_source": ("stage5a3_controller_evidence_stream"),
            "perception_transport": ("controller_evidence_filesystem_sidecar"),
            "controller_synchronized_evidence": True,
            "perception_live_during_mission": (self.live_during_mission),
            "perception_in_motor_control_loop": False,
            "policy_controls_motors": False,
            "perception_can_stop_robot": False,
            "absence_of_detection_is_not_a_safety_claim": True,
            "warmup_ms": warmup_ms,
            "inference_count": inference_count,
            "processed_frame_names": sorted(self.processed_names),
            "annotated_frame_count": (annotated_count),
            "all_processed_frames_annotated": (annotated_count == inference_count),
            "total_detections": sum(self.class_counter.values()),
            "detected_class_diversity": len(self.class_counter),
            "class_counts": dict(sorted(self.class_counter.items())),
            "risk_counts": dict(sorted(self.risk_counter.items())),
            "mean_inference_ms": (mean_inference_ms),
            "p95_inference_ms": (p95_inference_ms),
            "maximum_inference_ms": (maximum_inference_ms),
            "failure_count": self.failure_count,
            "stop_signal_observed": (stop_signal_observed),
        }

    def run(
        self,
        *,
        stop_file: pathlib.Path,
        warmup_ms: float,
        poll_interval_seconds: float = 0.15,
        idle_after_stop_seconds: float = 3.0,
        timeout_seconds: float = 900.0,
    ) -> dict[str, Any]:
        stop_file = pathlib.Path(stop_file).resolve()

        started_at = time.monotonic()
        last_activity_at = started_at
        stop_signal_observed = False

        while True:
            now = time.monotonic()

            if now - started_at > timeout_seconds:
                raise TimeoutError("The Stage 5B3 sidecar exceeded its timeout.")

            stop_requested = stop_file.exists()

            if stop_requested:
                stop_signal_observed = True

            processed_count = self.process_available_frames(
                mission_active=(not stop_requested),
            )

            if processed_count > 0:
                last_activity_at = time.monotonic()

            if stop_requested and time.monotonic() - last_activity_at >= idle_after_stop_seconds:
                break

            time.sleep(poll_interval_seconds)

        for _ in range(4):
            self.process_available_frames(
                mission_active=False,
            )

            time.sleep(poll_interval_seconds)

        summary = self.build_summary(
            warmup_ms=warmup_ms,
            stop_signal_observed=(stop_signal_observed),
        )

        write_json_atomic(
            self.summary_path,
            summary,
        )

        return summary
