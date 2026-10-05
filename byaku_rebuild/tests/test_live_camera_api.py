from collections import deque

import numpy as np
from fastapi.testclient import TestClient

import byaku_rebuild.api as api_module
import byaku_rebuild.pipeline_orchestrator as pipeline_module
from byaku_rebuild.pipeline_orchestrator import PipelineOrchestrator


class FakeCapture:
    def __init__(self, reads, *, opened=True):
        self.reads = deque(reads)
        self.opened = opened
        self.released = False
        self.read_calls = 0

    def isOpened(self):
        return self.opened

    def read(self):
        self.read_calls += 1
        if not self.reads:
            return False, None
        result = self.reads.popleft()
        if isinstance(result, Exception):
            raise result
        return result

    def release(self):
        self.released = True


def camera_frame(value=0):
    return np.full((16, 24, 3), value, dtype=np.uint8)


class FakeDetectionAdapter:
    def __init__(self):
        self._seen_frame_ids = set()
        self._last_frame_number = None
        self.calls = []

    def process_frame(self, frame, *, frame_id, frame_number):
        self.calls.append((frame_id, frame_number))
        self._seen_frame_ids.add(frame_id)
        self._last_frame_number = frame_number
        detection = {
            "track_id": 12,
            "class_name": "bottle",
            "confidence": 0.9,
            "bounding_box": [2.0, 2.0, 12.0, 12.0],
        }
        return {
            "status": "success",
            "orchestrator_detections": [detection],
            "warnings": [],
        }


class FakeDepthAdapter:
    def __init__(self):
        self.fail = False

    def estimate_depth(self, frame):
        if self.fail:
            return {
                "status": "error",
                "depth_available": False,
                "depth_map": None,
                "warnings": ["synthetic depth failure"],
            }
        height, width = frame.shape[:2]
        return {
            "status": "success",
            "depth_available": True,
            "depth_map": np.linspace(0.0, 1.0, height * width, dtype=np.float32).reshape(height, width),
            "metadata": {"output_shape": (height, width), "depth_ordering": "higher_is_nearer"},
            "warnings": [],
        }


def configure_api(monkeypatch, capture_factory=None):
    app = api_module.app
    app.state.camera_producer = None
    app.state.orchestrator = PipelineOrchestrator()
    app.state.detection_adapter = FakeDetectionAdapter()
    app.state.depth_adapter = FakeDepthAdapter()
    monkeypatch.setattr(app.state, "camera_capture_factory", capture_factory, raising=False)
    return app


def test_start_capture_once_and_stop_returns_actual_pipeline_result(monkeypatch):
    capture = FakeCapture([(True, camera_frame()), (True, camera_frame(1))])
    opened_sources = []
    configure_api(monkeypatch, lambda source: opened_sources.append(source) or capture)
    client = TestClient(api_module.app)

    started = client.post("/api/camera/start", json={"source": "synthetic-camera"})
    first = client.post("/api/camera/capture")
    second = client.post("/api/camera/capture")
    stopped = client.post("/api/camera/stop")

    assert started.status_code == 200
    assert started.json()["status"] == "started"
    assert opened_sources == ["synthetic-camera"]
    assert first.status_code == second.status_code == 200
    first_body = first.json()
    second_body = second.json()
    assert first_body["type"] == "camera_capture_result"
    assert first_body["status"] == "success"
    assert first_body["frame_result"]["type"] == "frame_result"
    assert first_body["frame_result"]["objects"][0]["class_name"] == "bottle"
    assert first_body["frame_number"] == 1
    assert second_body["frame_number"] == 2
    assert first_body["frame_id"] != second_body["frame_id"]
    assert stopped.status_code == 200
    assert stopped.json()["status"] == "stopped"
    assert capture.released is True
    assert client.post("/api/camera/stop").json()["already_stopped"] is True


def test_camera_open_failure_returns_structured_error_and_no_capture(monkeypatch):
    capture = FakeCapture([], opened=False)
    configure_api(monkeypatch, lambda _source: capture)
    client = TestClient(api_module.app)

    response = client.post("/api/camera/start", json={"source": 3})

    assert response.status_code == 503
    assert response.json()["type"] == "error"
    assert response.json()["stage"] == "camera"
    assert response.json()["code"] == "camera_unavailable"
    assert capture.released is True


def test_repeated_start_stop_and_source_change_are_explicit(monkeypatch):
    captures = [FakeCapture([]), FakeCapture([])]
    opened_sources = []

    def capture_factory(source):
        opened_sources.append(source)
        return captures[len(opened_sources) - 1]

    configure_api(monkeypatch, capture_factory)
    client = TestClient(api_module.app)

    assert client.post("/api/camera/start", json={"source": 0}).json()["status"] == "started"
    assert client.post("/api/camera/start", json={"source": 0}).json()["status"] == "already_running"
    conflict = client.post("/api/camera/start", json={"source": 1})
    assert conflict.status_code == 409
    assert conflict.json()["code"] == "camera_already_running"
    assert client.post("/api/camera/stop").json()["already_stopped"] is False
    assert client.post("/api/camera/stop").json()["already_stopped"] is True
    assert client.post("/api/camera/start", json={"source": 1}).json()["status"] == "started"
    assert opened_sources == [0, 1]
    assert captures[0].released is True
    client.post("/api/camera/stop")
    assert captures[1].released is True


def test_read_failures_are_structured_and_release_after_threshold(monkeypatch):
    capture = FakeCapture([(False, None), (False, None), (False, None)])
    configure_api(monkeypatch, lambda _source: capture)
    client = TestClient(api_module.app)
    client.post("/api/camera/start")

    responses = [client.post("/api/camera/capture") for _ in range(3)]

    assert all(response.status_code == 503 for response in responses)
    assert all(response.json()["stage"] == "camera" for response in responses)
    assert all(response.json()["code"] == "camera_read_failed" for response in responses)
    assert responses[0].json()["recoverable"] is True
    assert responses[-1].json()["recoverable"] is False
    assert capture.released is True
    assert client.post("/api/camera/capture").status_code == 409


def test_camera_read_exception_returns_read_error_and_releases(monkeypatch):
    capture = FakeCapture([RuntimeError("camera read raised")])
    configure_api(monkeypatch, lambda _source: capture)
    client = TestClient(api_module.app)
    client.post("/api/camera/start")

    response = client.post("/api/camera/capture")

    assert response.status_code == 503
    assert response.json()["code"] == "camera_read_failed"
    assert "camera read raised" in response.json()["message"]
    assert capture.released is True


def test_processing_failure_preserves_scene_then_camera_can_retry(monkeypatch):
    capture = FakeCapture([(True, camera_frame()), (True, camera_frame(1)), (True, camera_frame(2))])
    app = configure_api(monkeypatch, lambda _source: capture)
    client = TestClient(api_module.app)
    client.post("/api/camera/start", json={"source": 0})
    first = client.post("/api/camera/capture")
    assert first.status_code == 200
    prior_query = app.state.orchestrator.query("Is the bottle currently visible?")

    app.state.depth_adapter.fail = True
    failed = client.post("/api/camera/capture")

    assert failed.status_code == 502
    assert failed.json()["type"] == "error"
    assert failed.json()["stage"] == "camera"
    assert failed.json()["code"] == "frame_processing_failed"
    assert app.state.orchestrator._latest_frame_number == first.json()["frame_number"]
    assert app.state.orchestrator.query("Is the bottle currently visible?") == prior_query
    assert capture.released is True

    app.state.depth_adapter.fail = False
    restarted = client.post("/api/camera/start", json={"source": 0})
    retried = client.post("/api/camera/capture")

    assert restarted.status_code == 200
    assert retried.status_code == 200
    assert retried.json()["frame_number"] > first.json()["frame_number"]
    assert retried.json()["frame_id"] != first.json()["frame_id"]
    assert app.state.orchestrator._latest_frame_number == retried.json()["frame_number"]
    client.post("/api/camera/stop")


def test_busy_guard_rejects_capture_without_reading_or_submitting(monkeypatch):
    capture = FakeCapture([(True, camera_frame())])
    configure_api(monkeypatch, lambda _source: capture)
    client = TestClient(api_module.app)
    client.post("/api/camera/start")

    api_module._camera_operation_lock.acquire()
    try:
        response = client.post("/api/camera/capture")
    finally:
        api_module._camera_operation_lock.release()

    assert response.status_code == 409
    assert response.json()["code"] == "camera_busy"
    assert capture.read_calls == 0
    assert api_module.app.state.detection_adapter.calls == []
    client.post("/api/camera/stop")


def test_application_shutdown_releases_active_camera(monkeypatch):
    capture = FakeCapture([])
    configure_api(monkeypatch, lambda _source: capture)

    with TestClient(api_module.app) as client:
        assert client.post("/api/camera/start", json={"source": "shutdown-test"}).status_code == 200
        assert capture.released is False

    assert capture.released is True
    assert api_module.app.state.camera_producer is None


def test_existing_health_frame_and_query_endpoints_remain_functional(monkeypatch):
    app = configure_api(monkeypatch)
    client = TestClient(api_module.app)
    assert client.get("/health").status_code == 200

    import base64
    import cv2

    success, encoded = cv2.imencode(".jpg", camera_frame())
    assert success
    frame_response = client.post(
        "/api/frame",
        json={
            "frame_id": "existing-frame-route",
            "frame_number": 1,
            "width": 24,
            "height": 16,
            "data": base64.b64encode(encoded.tobytes()).decode("ascii"),
        },
    )
    query_response = client.post("/api/query", json={"query": "Is the bottle currently visible?"})

    assert frame_response.status_code == 200
    assert query_response.status_code == 200
    assert query_response.json()["status"] == "visible"