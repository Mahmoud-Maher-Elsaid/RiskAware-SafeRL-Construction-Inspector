from __future__ import annotations

import pathlib
import time

import cv2
import numpy as np
import pytest

from riskaware_saferrl.live_perception import (
    LiveDetection,
    LivePerceptionResult,
    build_risk_summary,
)
from riskaware_saferrl.live_perception_sidecar import (
    LivePerceptionSidecar,
    percentile,
)


class FakeBackend:
    backend_name = "fake_backend"
    device = "cuda:0"
    model_sha256 = "a" * 64
    model_connected = True
    model_path = pathlib.Path("fake_model.pt")

    def infer(
        self,
        frame: np.ndarray,
        *,
        frame_id: int,
    ) -> LivePerceptionResult:
        detections = (
            LiveDetection(
                class_id=11,
                class_name="Person",
                normalized_class_name=("person"),
                confidence=0.9,
                xyxy=(
                    1.0,
                    2.0,
                    8.0,
                    10.0,
                ),
            ),
            LiveDetection(
                class_id=8,
                class_name="NO-Hardhat",
                normalized_class_name=("no-hardhat"),
                confidence=0.8,
                xyxy=(
                    2.0,
                    3.0,
                    7.0,
                    9.0,
                ),
            ),
        )

        return LivePerceptionResult(
            frame_id=frame_id,
            model_connected=True,
            backend=self.backend_name,
            device=self.device,
            model_sha256=(self.model_sha256),
            frame_width=frame.shape[1],
            frame_height=frame.shape[0],
            inference_ms=10.0,
            detections=detections,
            risk=build_risk_summary(detections),
        )


def test_percentile_handles_interpolation() -> None:
    assert (
        percentile(
            [10.0, 20.0],
            0.5,
        )
        == 15.0
    )


def test_percentile_empty_input_is_zero() -> None:
    assert (
        percentile(
            [],
            0.95,
        )
        == 0.0
    )


def test_percentile_rejects_invalid_quantile() -> None:
    with pytest.raises(
        ValueError,
        match="quantile",
    ):
        percentile(
            [10.0],
            1.5,
        )


def test_sidecar_processes_controller_frame_once(
    tmp_path: pathlib.Path,
) -> None:
    evidence_root = tmp_path / "evidence"
    output_root = tmp_path / "output"

    evidence_root.mkdir()

    image_path = evidence_root / "waypoint_center.png"

    frame = np.zeros(
        (40, 60, 3),
        dtype=np.uint8,
    )

    assert cv2.imwrite(
        str(image_path),
        frame,
    )

    sidecar = LivePerceptionSidecar(
        backend=FakeBackend(),
        evidence_root=evidence_root,
        output_root=output_root,
        not_before_ns=0,
    )

    assert (
        sidecar.process_available_frames(
            mission_active=True,
        )
        == 0
    )

    assert (
        sidecar.process_available_frames(
            mission_active=True,
        )
        == 1
    )

    assert (
        sidecar.process_available_frames(
            mission_active=True,
        )
        == 0
    )

    summary = sidecar.build_summary(
        warmup_ms=100.0,
        stop_signal_observed=True,
    )

    assert summary["cv_model_connected"]

    assert summary["cuda_inference_verified"]

    assert summary["inference_count"] == 1

    assert summary["annotated_frame_count"] == 1

    assert summary["total_detections"] == 2

    assert summary["class_counts"]["person"] == 1

    assert summary["class_counts"]["no-hardhat"] == 1

    assert summary["perception_live_during_mission"]


def test_sidecar_rejects_corrupt_image(
    tmp_path: pathlib.Path,
) -> None:
    evidence_root = tmp_path / "evidence"
    output_root = tmp_path / "output"

    evidence_root.mkdir()

    image_path = evidence_root / "corrupt.png"

    image_path.write_bytes(b"not-a-valid-png")

    sidecar = LivePerceptionSidecar(
        backend=FakeBackend(),
        evidence_root=evidence_root,
        output_root=output_root,
        not_before_ns=0,
    )

    assert (
        sidecar.process_available_frames(
            mission_active=True,
        )
        == 0
    )

    with pytest.raises(
        RuntimeError,
        match="Could not decode",
    ):
        sidecar.process_available_frames(
            mission_active=True,
        )


def test_old_frames_are_ignored(
    tmp_path: pathlib.Path,
) -> None:
    evidence_root = tmp_path / "evidence"
    output_root = tmp_path / "output"

    evidence_root.mkdir()

    image_path = evidence_root / "old.png"

    frame = np.zeros(
        (20, 20, 3),
        dtype=np.uint8,
    )

    assert cv2.imwrite(
        str(image_path),
        frame,
    )

    future_start_ns = time.time_ns() + 10_000_000_000

    sidecar = LivePerceptionSidecar(
        backend=FakeBackend(),
        evidence_root=evidence_root,
        output_root=output_root,
        not_before_ns=future_start_ns,
    )

    assert (
        sidecar.process_available_frames(
            mission_active=True,
        )
        == 0
    )

    assert (
        sidecar.process_available_frames(
            mission_active=True,
        )
        == 0
    )
