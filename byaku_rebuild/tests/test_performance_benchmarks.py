"""Step 24 performance and regression benchmark tests."""
import time
from pathlib import Path

import cv2
import numpy as np
import pytest

from byaku_rebuild.depth_integration import DepthAnythingV2Adapter
from byaku_rebuild.detection_tracking import DEFAULT_WEIGHTS, DetectionTrackingAdapter
from byaku_rebuild.pipeline_orchestrator import PipelineOrchestrator


WORKSPACE_ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def loaded_models():
    depth_adapter = DepthAnythingV2Adapter(device="cpu")
    det_adapter = DetectionTrackingAdapter(weights_path=DEFAULT_WEIGHTS)
    orchestrator = PipelineOrchestrator()
    return depth_adapter, det_adapter, orchestrator


def test_yolo_bytetrack_latency_is_bounded(loaded_models):
    _, det_adapter, _ = loaded_models
    img_path = WORKSPACE_ROOT / "test1.jpg"
    frame = cv2.imread(str(img_path))
    assert frame is not None

    # Warm-up
    _ = det_adapter.process_frame(frame, frame_id="perf-warmup", frame_number=9001)

    # Measure steady-state
    t0 = time.perf_counter()
    result = det_adapter.process_frame(frame, frame_id="perf-test-1", frame_number=9002)
    latency_ms = (time.perf_counter() - t0) * 1000

    assert result["status"] == "success"
    assert len(result["detections"]) > 0
    # YOLO on CPU should easily be under 500ms
    assert latency_ms < 500.0


def test_depth_estimation_latency_is_measured_and_bounded(loaded_models):
    depth_adapter, _, _ = loaded_models
    img_path = WORKSPACE_ROOT / "test1.jpg"
    frame = cv2.imread(str(img_path))
    assert frame is not None

    # Warm-up
    _ = depth_adapter.estimate_depth(frame)

    t0 = time.perf_counter()
    result = depth_adapter.estimate_depth(frame)
    latency_ms = (time.perf_counter() - t0) * 1000

    assert result["status"] == "success"
    assert result["depth_available"] is True
    assert result["inference_time_ms"] is not None
    assert result["inference_time_ms"] > 0
    # Total depth call must complete within 3.5s on CPU
    assert latency_ms < 3500.0


def test_query_reasoning_latency_is_sub_50ms(loaded_models):
    depth_adapter, det_adapter, orchestrator = loaded_models
    img_path = WORKSPACE_ROOT / "test1.jpg"
    frame = cv2.imread(str(img_path))
    h, w = frame.shape[:2]

    det_res = det_adapter.process_frame(frame, frame_id="perf-q-1", frame_number=9101)
    depth_res = depth_adapter.estimate_depth(frame)

    _ = orchestrator.process_frame(
        frame_id="perf-q-1",
        frame_number=9101,
        image_dimensions={"width": w, "height": h},
        detections=det_res["orchestrator_detections"],
        depth_map=depth_res["depth_map"],
        timestamp=time.time(),
    )

    t0 = time.perf_counter()
    q_res = orchestrator.query("Where is the bed?")
    latency_ms = (time.perf_counter() - t0) * 1000

    assert q_res["status"] == "visible"
    assert "bed" in q_res["speech_response"]["spoken_text"].lower()
    assert latency_ms < 50.0, f"Query took too long: {latency_ms:.2f}ms"
