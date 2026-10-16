import base64
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

import Backend.fastapi_application as api_module
from Backend.fastapi_application import app
import Workflow.scene_pipeline as pipeline_module
from Workflow.scene_pipeline import PipelineOrchestrator


client = TestClient(app)


def _encode_frame(frame: np.ndarray) -> str:
    success, encoded = cv2.imencode(".jpg", frame)
    assert success is True
    return base64.b64encode(encoded.tobytes()).decode("ascii")


def _fake_detection_adapter() -> object:
    class FakeDetectionAdapter:
        def process_frame(self, frame, *, frame_id, frame_number):
            return {
                "status": "success",
                "frame": {
                    "frame_id": frame_id,
                    "frame_number": frame_number,
                    "image_shape": frame.shape[:2],
                    "width": frame.shape[1],
                    "height": frame.shape[0],
                },
                "detections": [{
                    "class_name": "bottle",
                    "confidence": 0.91,
                    "bounding_box": [1.0, 1.0, 8.0, 8.0],
                }],
                "tracks": [{
                    "track_id": 7,
                    "class_name": "bottle",
                    "confidence": 0.91,
                    "bounding_box": [1.0, 1.0, 8.0, 8.0],
                }],
                "orchestrator_detections": [{
                    "track_id": 7,
                    "class_name": "bottle",
                    "confidence": 0.91,
                    "bounding_box": [1.0, 1.0, 8.0, 8.0],
                }],
                "warnings": [],
            }

    return FakeDetectionAdapter()


def _fake_depth_adapter() -> object:
    class FakeDepthAdapter:
        def estimate_depth(self, frame):
            height, width = frame.shape[:2]
            map_values = np.linspace(0.0, 1.0, height * width, dtype=np.float32).reshape((height, width))
            return {
                "status": "success",
                "depth_available": True,
                "depth_map": map_values,
                "metadata": {"output_shape": (height, width), "depth_ordering": "higher_is_nearer"},
                "inference_time_ms": 1.0,
                "warnings": [],
            }

    return FakeDepthAdapter()


def _depth_adapter_with_result(result: dict) -> object:
    class FixedDepthAdapter:
        def estimate_depth(self, _frame):
            return result

    return FixedDepthAdapter()


def test_health_identifies_running_rebuild_api() -> None:
    response = client.get("/health")

    assert response.status_code == 200
    payload = response.json()
    assert payload["service"] == "byaku-rebuild-api"
    assert payload["status"] == "running"
    assert set(payload) == {"service", "status"}


def test_root_describes_live_api_capabilities() -> None:
    response = client.get("/")

    assert response.status_code == 200
    payload = response.json()
    assert payload["service"] == "BYAKU Rebuild API"
    assert "camera capture API" in payload["description"]


def test_unknown_route_returns_standard_not_found_error() -> None:
    response = client.get("/does-not-exist")

    assert response.status_code == 404
    assert response.json() == {
        "type": "error",
        "stage": "api",
        "code": "not_found",
        "message": "Not Found",
        "recoverable": True,
    }


def test_unsupported_method_returns_standard_method_not_allowed_error() -> None:
    response = client.post("/health", json={"unexpected": "data"})

    assert response.status_code == 405
    assert response.json() == {
        "type": "error",
        "stage": "api",
        "code": "method_not_allowed",
        "message": "Method Not Allowed",
        "recoverable": True,
    }


def test_valid_frame_request_processes_through_existing_pipeline(monkeypatch) -> None:
    frame = np.zeros((20, 40, 3), dtype=np.uint8)
    payload = {
        "frame_id": "api-frame-1",
        "frame_number": 1,
        "width": 40,
        "height": 20,
        "data": _encode_frame(frame),
        "timestamp": 1.0,
    }
    app.state.orchestrator = PipelineOrchestrator()
    app.state.detection_adapter = _fake_detection_adapter()
    app.state.depth_adapter = _fake_depth_adapter()

    response = client.post("/api/frame", json=payload)

    assert response.status_code == 200
    body = response.json()
    assert body["type"] == "frame_result"
    assert body["frame_size"] == {"width": 40, "height": 20}
    assert body["status"] == "success"
    assert body["objects"][0]["object_id"] == "track:7"
    assert "bottle" in body["scene_description"]["description"]
    assert body["stage_status"]["depth"] == "ready"
    depth_preview = cv2.imdecode(
        np.frombuffer(base64.b64decode(body["depth_preview"]), dtype=np.uint8),
        cv2.IMREAD_COLOR,
    )
    assert depth_preview.shape == (20, 40, 3)
    assert np.unique(depth_preview.reshape(-1, 3), axis=0).shape[0] > 1
    far_pixel, mid_pixel, near_pixel = (
        depth_preview[0, 3],
        depth_preview[10, 0],
        depth_preview[19, 37],
    )
    assert far_pixel[0] > far_pixel[2]
    assert mid_pixel[0] < 80 and mid_pixel[1] > 180 and mid_pixel[2] > 180
    assert near_pixel[2] > near_pixel[0]
    assert np.isfinite(body["objects"][0]["relative_depth_value"])
    assert body["objects"][0]["depth_category"] == "FAR"
    assert app.state.orchestrator.query("Is the bottle currently visible?")["status"] == "visible"


def test_query_route_uses_latest_valid_scene(monkeypatch) -> None:
    frame = np.zeros((20, 40, 3), dtype=np.uint8)
    app.state.orchestrator = PipelineOrchestrator()
    app.state.detection_adapter = _fake_detection_adapter()
    app.state.depth_adapter = _fake_depth_adapter()
    valid = client.post(
        "/api/frame",
        json={
            "frame_id": "api-frame-query",
            "frame_number": 1,
            "width": 40,
            "height": 20,
            "data": _encode_frame(frame),
            "timestamp": 5.0,
        },
    )
    assert valid.status_code == 200

    response = client.post("/api/query", json={"query": "Is the bottle currently visible?"})

    assert response.status_code == 200
    body = response.json()
    assert body["type"] == "query_result"
    assert body["status"] == "visible"
    assert "bottle" in body["answer"]
    assert body["speech_response"]["source_status"] == "visible"


def test_query_marks_old_scene_stale_and_reports_its_frame(monkeypatch) -> None:
    frame = np.zeros((20, 40, 3), dtype=np.uint8)
    app.state.orchestrator = PipelineOrchestrator()
    app.state.detection_adapter = _fake_detection_adapter()
    app.state.depth_adapter = _fake_depth_adapter()
    processed = client.post(
        "/api/frame",
        json={
            "frame_id": "api-frame-stale-query",
            "frame_number": 1,
            "width": 40,
            "height": 20,
            "data": _encode_frame(frame),
        },
    )
    assert processed.status_code == 200
    committed_at = app.state.orchestrator._latest_commit_wall_time
    assert isinstance(committed_at, float)
    monkeypatch.setattr(api_module.time, "monotonic", lambda: committed_at + 45)

    response = client.post("/api/query", json={"query": "Is the bottle currently visible?"})

    assert response.status_code == 200
    body = response.json()
    assert body["frame_id"] == "api-frame-stale-query"
    assert body["frame_number"] == 1
    assert any("may not reflect the current view" in warning for warning in body["warnings"])


def test_frame_status_does_not_claim_query_or_browser_tts_ready() -> None:
    app.state.orchestrator = PipelineOrchestrator()
    app.state.detection_adapter = _fake_detection_adapter()
    app.state.depth_adapter = _fake_depth_adapter()
    app.state.latest_query_response = None
    app.state.latest_stage_status = None
    frame = np.zeros((20, 40, 3), dtype=np.uint8)

    response = client.post(
        "/api/frame",
        json={
            "frame_id": "cold-start-status",
            "frame_number": 1,
            "width": 40,
            "height": 20,
            "data": _encode_frame(frame),
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["stage_status"]["reasoning"] == "unavailable"
    assert body["stage_status"]["tts"] == "unavailable"
    assert body["timing_source"] == "server_perf_counter"
    assert body["initialization_ms"] == 0.0
    assert body["processing_ms"] >= 0


def test_successful_query_reasoning_status_survives_a_new_frame() -> None:
    app.state.orchestrator = PipelineOrchestrator()
    app.state.detection_adapter = _fake_detection_adapter()
    app.state.depth_adapter = _fake_depth_adapter()
    app.state.latest_query_response = None
    app.state.latest_stage_status = None
    frame = np.zeros((20, 40, 3), dtype=np.uint8)
    encoded_frame = _encode_frame(frame)

    first_frame = client.post(
        "/api/frame",
        json={
            "frame_id": "query-status-frame-1",
            "frame_number": 1,
            "width": 40,
            "height": 20,
            "data": encoded_frame,
        },
    )
    assert first_frame.status_code == 200

    query = client.post("/api/query", json={"query": "Is the bottle currently visible?"})
    assert query.status_code == 200
    assert query.json()["status"] == "visible"
    assert app.state.latest_stage_status["reasoning"] == "ready"

    next_frame = client.post(
        "/api/frame",
        json={
            "frame_id": "query-status-frame-2",
            "frame_number": 2,
            "width": 40,
            "height": 20,
            "data": encoded_frame,
        },
    )
    assert next_frame.status_code == 200
    assert next_frame.json()["stage_status"]["reasoning"] == "ready"
    assert app.state.latest_stage_status["reasoning"] == "ready"


def test_status_reports_disconnected_camera_and_retained_last_frame(monkeypatch) -> None:
    monkeypatch.setattr(app.state, "camera_producer", None, raising=False)
    monkeypatch.setattr(app.state, "websocket_sessions", {}, raising=False)
    monkeypatch.setattr(
        app.state,
        "latest_frame_result",
        {"type": "frame_result", "frame_id": "retained-frame", "stage_status": {}},
        raising=False,
    )
    monkeypatch.setattr(
        app.state,
        "latest_frame_preview_data_url",
        "data:image/jpeg;base64,ZmFrZQ==",
        raising=False,
    )
    monkeypatch.setattr(
        app.state,
        "latest_stage_status",
        {"reasoning": "ready"},
        raising=False,
    )

    body = client.get("/api/status").json()

    assert body["camera"] == "disconnected"
    assert body["last_frame_available"] is True
    assert body["last_frame"]["frame_id"] == "retained-frame"
    assert body["last_frame"]["stage_status"]["camera"] == "disconnected"
    assert body["last_frame_preview"] == "data:image/jpeg;base64,ZmFrZQ=="


def test_frame_timing_separates_adapter_initialization_from_processing(monkeypatch) -> None:
    frame = np.zeros((20, 40, 3), dtype=np.uint8)
    app.state.orchestrator = None
    app.state.detection_adapter = None
    app.state.depth_adapter = None
    app.state.latest_query_response = None
    app.state.latest_stage_status = None

    def delayed_initialization():
        time.sleep(0.05)
        app.state.orchestrator = PipelineOrchestrator()
        app.state.detection_adapter = _fake_detection_adapter()
        app.state.depth_adapter = _fake_depth_adapter()
        return app.state.detection_adapter, app.state.depth_adapter, app.state.orchestrator

    monkeypatch.setattr(api_module, "_ensure_live_adapters", delayed_initialization)

    response = client.post(
        "/api/frame",
        json={
            "frame_id": "timing-separation-frame",
            "frame_number": 1,
            "width": 40,
            "height": 20,
            "data": _encode_frame(frame),
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["initialization_ms"] >= 40
    assert body["processing_ms"] < body["initialization_ms"]
    assert body["timing_source"] == "server_perf_counter"
    status = client.get("/api/status").json()
    assert status["initialization_ms"] == body["initialization_ms"]
    assert status["processing_ms"] == body["processing_ms"]
    assert status["timing_source"] == "server_perf_counter"


def test_depth_failure_preserves_scene_and_allows_frame_retry() -> None:
    frame = np.zeros((20, 40, 3), dtype=np.uint8)
    app.state.orchestrator = PipelineOrchestrator()
    detection_adapter = _fake_detection_adapter()
    app.state.detection_adapter = detection_adapter
    app.state.depth_adapter = _fake_depth_adapter()
    first = client.post(
        "/api/frame",
        json={
            "frame_id": "depth-safety-first",
            "frame_number": 1,
            "width": 40,
            "height": 20,
            "data": _encode_frame(frame),
        },
    )
    assert first.status_code == 200
    first_query = client.post("/api/query", json={"query": "Is the bottle currently visible?"}).json()

    class CountingDetectionAdapter:
        calls = 0

        def process_frame(self, *args, **kwargs):
            self.calls += 1
            return detection_adapter.process_frame(*args, **kwargs)

    counting_detection_adapter = CountingDetectionAdapter()
    app.state.detection_adapter = counting_detection_adapter
    app.state.depth_adapter = _depth_adapter_with_result(
        {
            "status": "error",
            "depth_available": False,
            "depth_map": None,
            "warnings": ["synthetic depth inference failure"],
        }
    )
    payload = {
        "frame_id": "depth-safety-retry",
        "frame_number": 2,
        "width": 40,
        "height": 20,
        "data": _encode_frame(frame),
    }
    failed = client.post("/api/frame", json=payload)

    assert failed.status_code == 400
    assert "depth inference failed" in failed.json()["message"]
    assert counting_detection_adapter.calls == 0
    assert app.state.orchestrator._latest_frame_number == 1
    assert app.state.orchestrator._seen_frame_ids == {"depth-safety-first"}
    assert client.post("/api/query", json={"query": "Is the bottle currently visible?"}).json() == first_query

    app.state.depth_adapter = _fake_depth_adapter()
    retried = client.post("/api/frame", json=payload)
    assert retried.status_code == 200
    assert retried.json()["frame_number"] == 2
    assert app.state.orchestrator._latest_frame_id == "depth-safety-retry"


def test_downstream_failure_restores_detection_frame_history_and_allows_retry(monkeypatch) -> None:
    frame = np.zeros((20, 40, 3), dtype=np.uint8)
    app.state.orchestrator = PipelineOrchestrator()
    delegate = _fake_detection_adapter()

    class StatefulDetectionAdapter:
        def __init__(self):
            self._seen_frame_ids = set()
            self._last_frame_number = None

        def process_frame(self, frame, *, frame_id, frame_number):
            self._seen_frame_ids.add(frame_id)
            self._last_frame_number = frame_number
            return delegate.process_frame(frame, frame_id=frame_id, frame_number=frame_number)

    detection_adapter = StatefulDetectionAdapter()
    app.state.detection_adapter = detection_adapter
    app.state.depth_adapter = _fake_depth_adapter()
    first_payload = {
        "frame_id": "downstream-first",
        "frame_number": 1,
        "width": 40,
        "height": 20,
        "data": _encode_frame(frame),
    }
    assert client.post("/api/frame", json=first_payload).status_code == 200

    def fail_description(*args, **kwargs):
        raise RuntimeError("synthetic description failure")

    monkeypatch.setattr(pipeline_module, "describe_scene", fail_description)
    retry_payload = {**first_payload, "frame_id": "downstream-retry", "frame_number": 2}
    failed = client.post("/api/frame", json=retry_payload)

    assert failed.status_code == 400
    assert detection_adapter._seen_frame_ids == {"downstream-first"}
    assert detection_adapter._last_frame_number == 1
    assert app.state.orchestrator._latest_frame_id == "downstream-first"
    assert app.state.orchestrator._latest_frame_number == 1
    assert app.state.orchestrator.query("Is the bottle currently visible?")["status"] == "visible"

    monkeypatch.undo()
    retried = client.post("/api/frame", json=retry_payload)
    assert retried.status_code == 200
    assert detection_adapter._seen_frame_ids == {"downstream-first", "downstream-retry"}
    assert detection_adapter._last_frame_number == 2


@pytest.mark.parametrize(
    ("depth_map", "detail"),
    [
        (np.ones((10, 10), dtype=np.float32), "do not match frame dimensions"),
        (np.full((20, 40), np.nan, dtype=np.float32), "non-finite values"),
    ],
)
def test_invalid_depth_maps_are_rejected_before_state_changes(depth_map, detail) -> None:
    app.state.orchestrator = PipelineOrchestrator()
    detection_adapter = _fake_detection_adapter()

    class CountingDetectionAdapter:
        calls = 0

        def process_frame(self, *args, **kwargs):
            self.calls += 1
            return detection_adapter.process_frame(*args, **kwargs)

    counting_detection_adapter = CountingDetectionAdapter()
    app.state.detection_adapter = counting_detection_adapter
    app.state.depth_adapter = _depth_adapter_with_result(
        {"status": "success", "depth_available": True, "depth_map": depth_map, "warnings": []}
    )
    response = client.post(
        "/api/frame",
        json={
            "frame_id": "invalid-depth",
            "frame_number": 1,
            "width": 40,
            "height": 20,
            "data": _encode_frame(np.zeros((20, 40, 3), dtype=np.uint8)),
        },
    )

    assert response.status_code == 400
    assert detail in response.json()["message"]
    assert counting_detection_adapter.calls == 0
    assert app.state.orchestrator.query("Is the bottle currently visible?")["status"] == "not_ready"


def test_constant_depth_partial_result_is_explicitly_supported() -> None:
    app.state.orchestrator = PipelineOrchestrator()
    app.state.detection_adapter = _fake_detection_adapter()
    app.state.depth_adapter = _depth_adapter_with_result(
        {
            "status": "partial",
            "depth_available": False,
            "depth_map": None,
            "metadata": {"raw_min": 0.5, "raw_max": 0.5},
            "warnings": ["depth prediction is constant; relative depth is unavailable for this frame"],
        }
    )

    response = client.post(
        "/api/frame",
        json={
            "frame_id": "constant-depth",
            "frame_number": 1,
            "width": 40,
            "height": 20,
            "data": _encode_frame(np.zeros((20, 40, 3), dtype=np.uint8)),
        },
    )

    assert response.status_code == 200
    assert response.json()["status"] == "partial"
    assert any("depth prediction is constant" in warning for warning in response.json()["warnings"])
    assert all(item["relative_depth_value"] is None for item in response.json()["objects"])
    assert "depth_preview" not in response.json()


def test_malformed_frame_requests_are_rejected_without_mutating_memory(monkeypatch) -> None:
    app.state.orchestrator = PipelineOrchestrator()
    app.state.detection_adapter = _fake_detection_adapter()
    app.state.depth_adapter = _fake_depth_adapter()
    valid = client.post(
        "/api/frame",
        json={
            "frame_id": "api-frame-good",
            "frame_number": 1,
            "width": 40,
            "height": 20,
            "data": _encode_frame(np.zeros((20, 40, 3), dtype=np.uint8)),
            "timestamp": 1.0,
        },
    )
    assert valid.status_code == 200

    invalid_base64 = client.post(
        "/api/frame",
        json={
            "frame_id": "api-frame-bad",
            "frame_number": 2,
            "width": 40,
            "height": 20,
            "data": "not-valid-base64",
            "timestamp": 2.0,
        },
    )
    assert invalid_base64.status_code == 400
    assert invalid_base64.json()["code"] == "http_error"

    mismatch = client.post(
        "/api/frame",
        json={
            "frame_id": "api-frame-mismatch",
            "frame_number": 3,
            "width": 20,
            "height": 40,
            "data": _encode_frame(np.zeros((20, 40, 3), dtype=np.uint8)),
            "timestamp": 3.0,
        },
    )
    assert mismatch.status_code == 400
    assert mismatch.json()["message"].startswith("frame dimensions")
    assert app.state.orchestrator.query("Is the bottle currently visible?")["status"] == "visible"


def test_repeated_and_out_of_order_frames_are_rejected_without_state_changes(monkeypatch) -> None:
    app.state.orchestrator = PipelineOrchestrator()
    app.state.detection_adapter = _fake_detection_adapter()
    app.state.depth_adapter = _fake_depth_adapter()

    first = client.post(
        "/api/frame",
        json={
            "frame_id": "repeat-1",
            "frame_number": 1,
            "width": 40,
            "height": 20,
            "data": _encode_frame(np.zeros((20, 40, 3), dtype=np.uint8)),
            "timestamp": 1.0,
        },
    )
    assert first.status_code == 200

    repeated = client.post(
        "/api/frame",
        json={
            "frame_id": "repeat-1",
            "frame_number": 2,
            "width": 40,
            "height": 20,
            "data": _encode_frame(np.zeros((20, 40, 3), dtype=np.uint8)),
            "timestamp": 2.0,
        },
    )
    assert repeated.status_code == 409
    assert repeated.json()["code"] == "http_error"

    out_of_order = client.post(
        "/api/frame",
        json={
            "frame_id": "repeat-2",
            "frame_number": 1,
            "width": 40,
            "height": 20,
            "data": _encode_frame(np.zeros((20, 40, 3), dtype=np.uint8)),
            "timestamp": 3.0,
        },
    )
    assert out_of_order.status_code == 409
    assert "not greater than the latest processed" in out_of_order.json()["message"]
    assert app.state.orchestrator.query("Is the bottle currently visible?")["status"] == "visible"


def test_invalid_request_schema_returns_standard_api_validation_error() -> None:
    response = client.post("/api/frame", json={"frame_id": "", "frame_number": 0, "width": 40, "height": 20, "data": ""})

    assert response.status_code == 422
    payload = response.json()
    assert payload["type"] == "error"
    assert payload["stage"] == "api"
    assert payload["code"] == "invalid_request"
    assert payload["recoverable"] is True


def test_status_reports_only_known_state_before_processing(monkeypatch) -> None:
    for name, value in (
        ("camera_producer", None),
        ("websocket_sessions", {}),
        ("latest_frame_result", None),
        ("latest_query_response", None),
        ("latest_stage_status", None),
    ):
        monkeypatch.setattr(app.state, name, value, raising=False)

    response = client.get("/api/status")

    assert response.status_code == 200
    body = response.json()
    assert body["backend"] == "connected"
    assert body["camera"] == "disconnected"
    assert body["last_frame"] is None
    assert body["last_query"] is None
    assert body["last_narration"] is None
    assert body["stage_status"]["depth"] == "unavailable"
    assert "processing_ms" not in body
    assert "events" not in body
    assert "depth_preview" not in body


def test_image_upload_uses_vision_pipeline_and_returns_annotated_results(monkeypatch) -> None:
    monkeypatch.setattr(
        app.state,
        "upload_image_components_factory",
        lambda: (_fake_detection_adapter(), _fake_depth_adapter(), PipelineOrchestrator()),
        raising=False,
    )
    success, encoded = cv2.imencode(".png", np.zeros((20, 40, 3), dtype=np.uint8))
    assert success

    response = client.post(
        "/api/upload/image",
        files={"upload": ("sample.png", encoded.tobytes(), "image/png")},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["media_type"] == "image"
    assert body["status"] == "success"
    assert body["frame_size"] == {"width": 40, "height": 20}
    assert "bottle" in body["scene_description"]
    assert body["objects"][0]["track_id"] == 7
    assert body["objects"][0]["horizontal_position"] == "left"
    assert body["objects"][0]["depth_category"] == "FAR"
    annotated = cv2.imdecode(
        np.frombuffer(base64.b64decode(body["annotated_image"]), dtype=np.uint8),
        cv2.IMREAD_COLOR,
    )
    assert annotated.shape == (20, 40, 3)


def test_image_upload_reports_unknown_depth_when_estimation_fails(monkeypatch) -> None:
    class FailedDepthAdapter:
        def estimate_depth(self, _frame):
            raise RuntimeError("depth model unavailable")

    monkeypatch.setattr(
        app.state,
        "upload_image_components_factory",
        lambda: (_fake_detection_adapter(), FailedDepthAdapter(), PipelineOrchestrator()),
        raising=False,
    )
    success, encoded = cv2.imencode(".png", np.zeros((20, 40, 3), dtype=np.uint8))
    assert success

    response = client.post(
        "/api/upload/image",
        files={"upload": ("sample.png", encoded.tobytes(), "image/png")},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "partial"
    assert body["objects"][0]["depth_category"] == "UNKNOWN"
    assert any("depth is unavailable" in warning for warning in body["warnings"])


@pytest.mark.parametrize(
    ("filename", "content", "status_code"),
    [
        ("sample.gif", b"not an image", 415),
        ("sample.png", b"", 400),
        ("sample.jpg", b"not a jpeg", 415),
    ],
)
def test_invalid_image_uploads_are_rejected(filename, content, status_code) -> None:
    response = client.post(
        "/api/upload/image",
        files={"upload": (filename, content, "application/octet-stream")},
    )
    assert response.status_code == status_code
    assert "Traceback" not in response.text


def test_oversized_image_upload_is_rejected(monkeypatch) -> None:
    monkeypatch.setattr(api_module, "_UPLOAD_IMAGE_MAX_BYTES", 16)
    response = client.post(
        "/api/upload/image",
        files={"upload": ("large.png", b"x" * 17, "image/png")},
    )
    assert response.status_code == 413
    assert "16 bytes" in response.json()["message"]


def test_video_upload_uses_sampled_video_pipeline_and_cleans_temporary_file(monkeypatch) -> None:
    captured: dict[str, Any] = {}

    class FakeVideoPipeline:
        def process_video(self, path, *, max_frames, frame_stride):
            captured.update({
                "path": Path(path),
                "max_frames": max_frames,
                "frame_stride": frame_stride,
                "path_exists": Path(path).is_file(),
            })
            return {
                "final_status": "success",
                "frames_processed": 2,
                "frames_skipped": 58,
                "per_frame_results": [{
                    "frame": {"frame_number": 1, "timestamp_seconds": 0.0},
                    "objects": [{
                        "class_name": "person",
                        "track_id": 8,
                        "horizontal_position": "LEFT",
                        "depth_category": "NEAR",
                    }],
                    "scene_description": {"description": "A person is on the left and appears near."},
                }, {
                    "frame": {"frame_number": 31, "timestamp_seconds": 1.0},
                    "objects": [{
                        "class_name": "person",
                        "track_id": 8,
                        "horizontal_position": "CENTER",
                        "depth_category": "MID-RANGE",
                    }],
                    "scene_description": {"description": "A person is in the center at mid-range."},
                }],
            }

    monkeypatch.setattr(app.state, "upload_video_pipeline_factory", FakeVideoPipeline, raising=False)
    response = client.post(
        "/api/upload/video",
        files={"upload": ("walk.mp4", b"video bytes", "video/mp4")},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["media_type"] == "video"
    assert body["status"] == "success"
    assert body["frames_processed"] == 2
    assert body["frame_stride"] == 30
    assert len(body["samples"]) == 2
    assert body["objects"][0]["track_id"] == 8
    assert "Later" in body["scene_description"]
    assert captured["path_exists"] is True
    assert captured["max_frames"] == 20
    assert captured["frame_stride"] == 30
    assert not captured["path"].exists()


def test_invalid_video_codec_returns_readable_error_and_removes_upload(monkeypatch) -> None:
    captured_paths: list[Path] = []

    class UnsupportedVideoPipeline:
        def process_video(self, path, **_kwargs):
            captured_paths.append(Path(path))
            return {
                "final_status": "error",
                "frames_processed": 0,
                "warnings": [f"could not open video: {path}"],
            }

    monkeypatch.setattr(app.state, "upload_video_pipeline_factory", UnsupportedVideoPipeline, raising=False)
    response = client.post(
        "/api/upload/video",
        files={"upload": ("broken.webm", b"not a video", "video/webm")},
    )

    assert response.status_code == 422
    assert "supported codec" in response.json()["message"]
    assert str(captured_paths[0]) not in response.text
    assert not captured_paths[0].exists()


def test_empty_and_unsupported_video_uploads_are_rejected() -> None:
    empty = client.post(
        "/api/upload/video",
        files={"upload": ("empty.mp4", b"", "video/mp4")},
    )
    unsupported = client.post(
        "/api/upload/video",
        files={"upload": ("movie.m4v", b"video", "video/mp4")},
    )
    assert empty.status_code == 400
    assert "empty" in empty.json()["message"]
    assert unsupported.status_code == 415


def test_oversized_video_upload_is_rejected(monkeypatch) -> None:
    monkeypatch.setattr(api_module, "_UPLOAD_VIDEO_MAX_BYTES", 16)
    response = client.post(
        "/api/upload/video",
        files={"upload": ("large.mp4", b"x" * 17, "video/mp4")},
    )
    assert response.status_code == 413
    assert "16 bytes" in response.json()["message"]


def test_video_upload_provides_review_tokens_and_isolated_detection_ids(monkeypatch) -> None:
    class MockVideoPipeline:
        def process_video(self, path, *, max_frames, frame_stride):
            return {
                "final_status": "success",
                "frames_processed": 2,
                "frames_skipped": 0,
                "frames_failed": 0,
                "per_frame_results": [
                    {
                        "frame": {"frame_number": 1, "timestamp_seconds": 0.0},
                        "annotated_image": "YW5ub3RhdGVkMQ==",
                        "annotated_size": (640, 480),
                        "review_frame_image": "cmF3MQ==",
                        "objects": [
                            {"class_name": "person", "track_id": 5, "horizontal_position": "CENTER", "depth_category": "NEAR", "bounding_box": [10, 20, 100, 200]},
                            {"class_name": "chair", "track_id": None, "horizontal_position": "LEFT", "depth_category": "UNKNOWN", "bounding_box": [150, 50, 250, 220]},
                        ],
                        "scene_description": {"description": "A person near a chair."},
                    },
                ],
            }

    monkeypatch.setattr(app.state, "upload_video_pipeline_factory", MockVideoPipeline, raising=False)
    response = client.post(
        "/api/upload/video",
        files={"upload": ("walk.mp4", b"video-data", "video/mp4")},
    )
    assert response.status_code == 200
    data = response.json()
    session_id = data["session_id"]
    sample = data["samples"][0]

    assert sample["review_frame_token"] is not None
    assert sample["annotated_image"] == "YW5ub3RhdGVkMQ=="

    # Scoped tracked detection ID
    tracked_det = sample["objects"][0]
    assert tracked_det["track_id"] == 5
    assert tracked_det["detection_id"] == f"{session_id}:track:5"

    # Scoped untracked detection ID
    untracked_det = sample["objects"][1]
    assert untracked_det["track_id"] is None
    assert untracked_det["detection_id"] == f"{session_id}:frame-1:det-1"


def test_uploaded_video_and_live_feedback_remain_isolated_in_history(monkeypatch, tmp_path) -> None:
    from Backend.feedback_store import FeedbackStore
    store_path = tmp_path / "review" / "history.json"
    monkeypatch.setattr(api_module, "_feedback_store", FeedbackStore(store_path))

    # 1. Submit review for uploaded video detection with track_id=3
    video_feedback = client.post(
        "/api/feedback",
        json={
            "source": "upload_video",
            "session_id": "session-video-abc",
            "frame_number": 1,
            "detection_id": "session-video-abc:track:3",
            "track_id": 3,
            "original_prediction": {"class_name": "cat", "confidence": 0.89, "bounding_box": [5, 5, 50, 50]},
            "correction_type": "confirm",
            "review_status": "confirmed",
        },
    )
    assert video_feedback.status_code == 201
    v_body = video_feedback.json()
    assert v_body["source"] == "upload_video"
    assert v_body["session_id"] == "session-video-abc"
    assert v_body["detection_id"] == "session-video-abc:track:3"

    # 2. Submit review for live camera detection with same track_id=3
    live_feedback = client.post(
        "/api/feedback",
        json={
            "source": "live",
            "session_id": "session-live-xyz",
            "frame_number": 42,
            "detection_id": "track:3",
            "track_id": 3,
            "original_prediction": {"class_name": "dog", "confidence": 0.77, "bounding_box": [20, 20, 80, 80]},
            "correction_type": "class_correction",
            "review_status": "corrected",
            "human_corrected_name": "puppy",
        },
    )
    assert live_feedback.status_code == 201
    l_body = live_feedback.json()
    assert l_body["source"] == "live"
    assert l_body["session_id"] == "session-live-xyz"
    assert l_body["human_corrected_name"] == "puppy"

    # 3. Verify history distinguishes the two and they do not collide
    all_feedback = client.get("/api/feedback").json()["feedback"]
    assert len(all_feedback) == 2

    v_rec = next(r for r in all_feedback if r["source"] == "upload_video")
    l_rec = next(r for r in all_feedback if r["source"] == "live")

    assert v_rec["original_prediction"]["class_name"] == "cat"
    assert v_rec["review_status"] == "confirmed"
    assert l_rec["original_prediction"]["class_name"] == "dog"
    assert l_rec["human_corrected_name"] == "puppy"