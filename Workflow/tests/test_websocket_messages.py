import base64

import cv2
import numpy as np
from fastapi.testclient import TestClient

import Backend.fastapi_application as api_module
from Backend.frame_timing_profiler import FrameProfiler
from Workflow.scene_pipeline import PipelineOrchestrator


class FakeDetectionAdapter:
    def __init__(self):
        self.calls = []
        self._seen_frame_ids = set()
        self._last_frame_number = None

    def process_frame(self, frame, *, frame_id, frame_number):
        self.calls.append((frame_id, frame_number))
        self._seen_frame_ids.add(frame_id)
        self._last_frame_number = frame_number
        detection = {
            "track_id": 19,
            "class_name": "bottle",
            "confidence": 0.87,
            "bounding_box": [2.0, 2.0, 10.0, 12.0],
        }
        second_detection = {
            "track_id": 20,
            "class_name": "chair",
            "confidence": 0.82,
            "bounding_box": [14.0, 2.0, 22.0, 12.0],
        }
        return {
            "status": "success",
            "detections": [
                {key: value for key, value in item.items() if key != "track_id"}
                for item in (detection, second_detection)
            ],
            "tracks": [detection, second_detection],
            "orchestrator_detections": [detection, second_detection],
            "warnings": [],
        }


class FakeDepthAdapter:
    def __init__(self):
        self.fail = False
        self.calls = 0

    def estimate_depth(self, frame):
        self.calls += 1
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


class UnrelatedPythonProducer:
    def __init__(self, is_running=True):
        self.is_running = is_running
        self.start_calls = 0
        self.stop_calls = 0

    def start(self):
        self.start_calls += 1

    def stop(self):
        self.stop_calls += 1


def configure_api(monkeypatch):
    app = api_module.app
    app.state.orchestrator = PipelineOrchestrator()
    app.state.detection_adapter = FakeDetectionAdapter()
    app.state.depth_adapter = FakeDepthAdapter()
    app.state.camera_producer = None
    app.state.camera_capture_factory = lambda _source: (_ for _ in ()).throw(AssertionError("browser protocol opened Python camera"))
    app.state.websocket_sessions = {}
    app.state.last_assigned_frame_number = 0
    app.state.latest_frame_result = None
    app.state.latest_query_response = None
    app.state.latest_stage_status = None
    return app


def jpeg_payload(frame=None):
    frame = frame if frame is not None else np.zeros((16, 24, 3), dtype=np.uint8)
    success, encoded = cv2.imencode(".jpg", frame)
    assert success
    return base64.b64encode(encoded.tobytes()).decode("ascii")


def browser_frame(frame_id, *, data=None, width=24, height=16):
    return {
        "type": "frame",
        "frame_id": frame_id,
        "width": width,
        "height": height,
        "data": data if data is not None else jpeg_payload(),
    }


def start_browser_camera(websocket):
    websocket.send_json({"type": "start_camera"})
    response = websocket.receive_json()
    assert response["type"] == "status"
    assert response["stage_status"]["camera"] == "connected"
    return response


def test_websocket_start_stop_are_session_only_and_never_start_python_camera(monkeypatch):
    app = configure_api(monkeypatch)
    producer = UnrelatedPythonProducer(is_running=False)
    app.state.camera_producer = producer
    client = TestClient(app)

    with client.websocket_connect("/ws") as websocket:
        connected = websocket.receive_json()
        assert connected["type"] == "status"
        assert connected["backend"] == "connected"
        started = start_browser_camera(websocket)
        assert started["camera"] == "connected"
        websocket.send_json({"type": "stop_camera"})
        stopped = websocket.receive_json()
        assert stopped["type"] == "status"
        assert stopped["camera"] == "disconnected"

    assert producer.start_calls == 0
    assert producer.stop_calls == 0
    assert app.state.camera_producer is producer
    assert app.state.websocket_sessions == {}


def test_websocket_frames_echo_browser_id_and_assign_unique_monotonic_server_frames(monkeypatch):
    app = configure_api(monkeypatch)
    client = TestClient(app)
    first_internal_ids = []

    with client.websocket_connect("/ws") as websocket:
        websocket.receive_json()
        start_browser_camera(websocket)
        for client_frame_id in (1, 2):
            websocket.send_json(browser_frame(client_frame_id))
            response = websocket.receive_json()
            assert response["type"] == "frame_result"
            assert response["frame_id"] == client_frame_id
            assert response["frame_number"] == client_frame_id
            assert response["status"] == "success"
            assert response["objects"][0]["class_name"] == "bottle"
            assert response["tracks"][0]["track_id"] == 19
            assert response["scene_narration"]
            assert response["relationships"][0]["source"] == response["relationships"][0]["subject"]
            first_internal_ids.append(app.state.orchestrator._latest_frame_id)

    with client.websocket_connect("/ws") as websocket:
        websocket.receive_json()
        start_browser_camera(websocket)
        websocket.send_json(browser_frame(1))
        response = websocket.receive_json()

    assert response["frame_id"] == 1
    assert response["frame_number"] == 3
    assert app.state.detection_adapter.calls == [
        (first_internal_ids[0], 1),
        (first_internal_ids[1], 2),
        (app.state.orchestrator._latest_frame_id, 3),
    ]
    assert len(set(first_internal_ids + [app.state.orchestrator._latest_frame_id])) == 3
    assert all(frame_id.startswith("browser:") for frame_id in first_internal_ids)


def test_frame_profiler_records_live_websocket_stages(monkeypatch):
    app = configure_api(monkeypatch)
    profiler = FrameProfiler()
    monkeypatch.setattr(app.state, "frame_profiler", profiler, raising=False)
    client = TestClient(app)

    with client.websocket_connect("/ws") as websocket:
        websocket.receive_json()
        start_browser_camera(websocket)
        websocket.send_json(browser_frame(1))
        response = websocket.receive_json()

    assert response["type"] == "frame_result"
    stages = profiler.summary()["stages"]
    for stage in (
        "websocket_receive",
        "websocket_json_parse",
        "frame_request_validation",
        "adapter_lookup_or_initialization",
        "jpeg_base64_decode_and_opencv_decode",
        "depth_anything_total",
        "yolo_bytetrack_total",
        "spatial_relationships",
        "scene_memory",
        "scene_graph",
        "scene_description",
        "scene_narration",
        "response_payload_preparation",
        "response_json_serialization",
        "websocket_send",
    ):
        assert stages[stage]["count"] == 1


def test_websocket_query_maps_actual_result_and_speech_text_to_narration(monkeypatch):
    app = configure_api(monkeypatch)
    client = TestClient(app)

    with client.websocket_connect("/ws") as websocket:
        websocket.receive_json()
        start_browser_camera(websocket)
        websocket.send_json(browser_frame(1))
        frame_result = websocket.receive_json()
        websocket.send_json({"type": "query", "text": "Is the bottle currently visible?"})
        query_result = websocket.receive_json()

    assert frame_result["status"] == "success"
    assert query_result["type"] == "query_result"
    assert query_result["status"] == "visible"
    assert "bottle" in query_result["answer"]
    assert query_result["narration"] == query_result["speech_response"]["spoken_text"]
    assert query_result["reasoning"]["objects"] == query_result["matched_objects"]
    assert "audio" not in query_result


def test_websocket_ping_malformed_json_and_unsupported_type_are_explicit(monkeypatch):
    configure_api(monkeypatch)
    client = TestClient(api_module.app)

    with client.websocket_connect("/ws") as websocket:
        websocket.receive_json()
        websocket.send_json({"type": "ping", "ts": 42})
        assert websocket.receive_json() == {"type": "pong", "ts": 42}
        websocket.send_text("{")
        malformed = websocket.receive_json()
        assert malformed["type"] == "error"
        assert malformed["stage"] == "protocol"
        assert malformed["code"] == "invalid_json"
        websocket.send_json({"type": "not_supported"})
        unsupported = websocket.receive_json()
        assert unsupported["code"] == "unsupported_message"
        assert unsupported["stage"] == "protocol"


def test_websocket_invalid_frame_and_inference_error_preserve_scene_and_allow_retry(monkeypatch):
    app = configure_api(monkeypatch)
    client = TestClient(app)

    with client.websocket_connect("/ws") as websocket:
        websocket.receive_json()
        start_browser_camera(websocket)
        websocket.send_json(browser_frame(1))
        first = websocket.receive_json()
        previous_query = app.state.orchestrator.query("Is the bottle currently visible?")

        websocket.send_json(browser_frame(2, data="not-base64"))
        malformed = websocket.receive_json()
        assert malformed["type"] == "error"
        assert malformed["code"] == "invalid_frame"
        assert malformed["frame_id"] == 2
        assert app.state.orchestrator._latest_frame_number == first["frame_number"]

        websocket.send_json(browser_frame(3, width=23))
        dimensions = websocket.receive_json()
        assert dimensions["type"] == "error"
        assert dimensions["code"] == "invalid_frame"
        assert "dimensions" in dimensions["message"]
        assert app.state.orchestrator._latest_frame_number == first["frame_number"]

        app.state.depth_adapter.fail = True
        websocket.send_json(browser_frame(4))
        failed = websocket.receive_json()
        assert failed["type"] == "error"
        assert failed["code"] == "frame_processing_failed"
        assert failed["frame_id"] == 4
        assert app.state.orchestrator._latest_frame_number == first["frame_number"]
        assert app.state.orchestrator.query("Is the bottle currently visible?") == previous_query

        app.state.depth_adapter.fail = False
        websocket.send_json(browser_frame(5))
        retried = websocket.receive_json()

    assert retried["type"] == "frame_result"
    assert retried["frame_id"] == 5
    assert retried["frame_number"] > first["frame_number"]
    assert app.state.orchestrator._latest_frame_id.startswith("browser:")


def test_websocket_overlap_is_rejected_without_duplicate_processing(monkeypatch):
    app = configure_api(monkeypatch)
    client = TestClient(app)

    with client.websocket_connect("/ws") as websocket:
        websocket.receive_json()
        start_browser_camera(websocket)
        assert api_module._frame_processing_lock.acquire(blocking=False)
        try:
            websocket.send_json(browser_frame(1))
            busy = websocket.receive_json()
        finally:
            api_module._frame_processing_lock.release()
        assert busy["type"] == "error"
        assert busy["code"] == "frame_processing_busy"
        assert app.state.detection_adapter.calls == []

        websocket.send_json(browser_frame(1))
        accepted = websocket.receive_json()

    assert accepted["type"] == "frame_result"
    assert accepted["frame_number"] == 1
    assert len(app.state.detection_adapter.calls) == 1


def test_disconnect_preserves_scene_and_does_not_stop_python_camera(monkeypatch):
    app = configure_api(monkeypatch)
    client = TestClient(app)
    valid_request = api_module.FrameRequest(
        frame_id="rest-before-disconnect",
        frame_number=1,
        width=24,
        height=16,
        data=jpeg_payload(),
    )
    committed = api_module.process_live_frame(valid_request)
    previous_query = app.state.orchestrator.query("Is the bottle currently visible?")
    producer = UnrelatedPythonProducer(is_running=True)
    app.state.camera_producer = producer

    with client.websocket_connect("/ws") as websocket:
        websocket.receive_json()
        websocket.send_json({"type": "stop_camera"})
        assert websocket.receive_json()["type"] == "status"

    assert producer.stop_calls == 0
    assert app.state.orchestrator._latest_frame_number == committed["frame_number"]
    assert app.state.orchestrator.query("Is the bottle currently visible?") == previous_query
    assert app.state.websocket_sessions == {}


def test_api_status_reports_latest_actual_frame_and_query_without_invented_metrics(monkeypatch):
    app = configure_api(monkeypatch)
    client = TestClient(app)
    before = client.get("/api/status").json()
    assert before["backend"] == "connected"
    assert before["camera"] == "disconnected"
    assert before["last_frame"] is None
    assert before["stage_status"]["depth"] == "unavailable"

    frame = client.post(
        "/api/frame",
        json={
            "frame_id": "status-frame",
            "frame_number": 1,
            "width": 24,
            "height": 16,
            "data": jpeg_payload(),
        },
    )
    assert frame.status_code == 200
    query = client.post("/api/query", json={"query": "Is the bottle currently visible?"})
    assert query.status_code == 200

    status = client.get("/api/status").json()
    assert status["last_frame"]["frame_id"] == "status-frame"
    assert status["last_frame"]["objects"][0]["class_name"] == "bottle"
    assert status["last_query"] == "Is the bottle currently visible?"
    assert status["last_reasoning"]["objects"] == query.json()["matched_objects"]
    assert isinstance(status.get("processing_ms"), (int, float)) and status["processing_ms"] >= 0
    assert "processing_fps" not in status
    assert "events" not in status
    assert "depth_preview" not in status


def test_websocket_start_refuses_active_python_producer_without_touching_it(monkeypatch):
    app = configure_api(monkeypatch)
    producer = UnrelatedPythonProducer(is_running=True)
    app.state.camera_producer = producer
    client = TestClient(app)

    with client.websocket_connect("/ws") as websocket:
        websocket.receive_json()
        websocket.send_json({"type": "start_camera"})
        response = websocket.receive_json()

    assert response["type"] == "error"
    assert response["code"] == "camera_busy"
    assert producer.start_calls == producer.stop_calls == 0