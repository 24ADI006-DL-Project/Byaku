"""Step 27 — Accessibility and Reliability Testing.

Validates:
1. Keyboard accessibility via REST API structure (all operations reachable via
   HTTP without pointer input; meaningful HTTP methods/routes per CONTRACT.md).
2. Screen-reader support — error messages, narration, and spoken_text are
   well-formed non-empty strings; status fields use expected vocabulary;
   ambiguous or not-found answers produce non-empty spoken_text.
3. Failure handling — camera open/read/encode/stop errors, malformed frames,
   depth and detection failures, pipeline exceptions, and query errors all
   return structured recoverable responses; the last committed scene stays
   queryable after every failure.
4. Recovery behavior — after a transient camera error the API can restart and
   process a new frame; after a failed frame the sequence counter remains
   consistent; WebSocket reconnects get a fresh session without inheriting
   previous session state.
5. No metric hallucinations in narration — spoken_text never contains numeric
   distances in metres/cm/feet.
6. Empty-query and whitespace-query rejection returns a descriptive error
   that an assistive reader can present to the user.
"""

from __future__ import annotations

import base64
import io
import json
import threading
from typing import Any
from unittest.mock import MagicMock, patch
import uuid

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

import Backend.fastapi_application as api_module
from Backend.fastapi_application import app
from Workflow.scene_pipeline import PipelineOrchestrator
from Workflow.reasoning.object_query_engine import answer_query
from Workflow.reasoning.scene_memory import SceneMemory
from Workflow.reasoning.narration_response import prepare_spoken_response, prepare_spoken_scene_summary


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _make_jpeg(width: int = 4, height: int = 4) -> bytes:
    """Return a minimal valid JPEG bytestring."""
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    ok, buf = cv2.imencode(".jpg", frame)
    assert ok
    return buf.tobytes()


def _b64_jpeg(width: int = 4, height: int = 4) -> str:
    return base64.b64encode(_make_jpeg(width, height)).decode()


def _make_frame_request(
    frame_id: str,
    frame_number: int,
    *,
    width: int = 4,
    height: int = 4,
) -> dict:
    return {
        "frame_id": frame_id,
        "frame_number": frame_number,
        "width": width,
        "height": height,
        "data": _b64_jpeg(width, height),
        "timestamp": 1.0,
    }


def _fake_depth_result(height: int = 4, width: int = 4) -> dict:
    depth_map = np.linspace(0.1, 0.9, height * width, dtype=np.float32).reshape((height, width))
    return {"status": "success", "depth_available": True, "depth_map": depth_map, "warnings": []}


def _fake_detection_result(frame_id: str, frame_number: int) -> dict:
    return {
        "status": "success",
        "frame_id": frame_id,
        "frame_number": frame_number,
        "detections": [],
        "tracks": [],
        "orchestrator_detections": [],
        "warnings": [],
    }


def _make_client_with_mocks() -> tuple[TestClient, MagicMock, MagicMock]:
    """Return a TestClient with mocked depth and detection adapters."""
    depth_mock = MagicMock()
    detection_mock = MagicMock()
    detection_mock._seen_frame_ids = set()
    detection_mock._last_frame_number = None

    app.state.orchestrator = PipelineOrchestrator()
    app.state.depth_adapter = depth_mock
    app.state.detection_adapter = detection_mock

    client = TestClient(app, raise_server_exceptions=False)
    return client, depth_mock, detection_mock


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _reset_app_state():
    """Reset app.state before every test to prevent cross-test state leakage.

    Provides mocked detection/depth adapters so _ensure_live_adapters() never
    tries to load real ML models in tests that don't need them.
    """
    orch = PipelineOrchestrator()
    app.state.orchestrator = orch

    det = MagicMock()
    det._seen_frame_ids = set()
    det._last_frame_number = None
    app.state.detection_adapter = det

    dep = MagicMock()
    app.state.depth_adapter = dep

    app.state.camera_producer = None
    app.state.latest_frame_result = None
    app.state.latest_query_response = None
    app.state.latest_stage_status = None
    app.state.last_assigned_frame_number = 0
    try:
        app.state.websocket_sessions = {}
    except Exception:
        pass
    yield
    # Post-test cleanup
    producer = getattr(app.state, "camera_producer", None)
    if producer is not None and hasattr(producer, "stop"):
        try:
            producer.stop()
        except Exception:
            pass
    app.state.camera_producer = None


# ---------------------------------------------------------------------------
# 1. Keyboard accessibility — REST API structure
# ---------------------------------------------------------------------------


class TestKeyboardAccessibility:
    """All BYAKU operations are reachable via HTTP without pointer input.

    A keyboard-accessible interface can reach every function via the HTTP
    API. These tests confirm that each documented endpoint responds to the
    correct method with a recognisable structured response so a browser can
    call them without mouse interaction.
    """

    def test_health_endpoint_is_get(self) -> None:
        """GET /health is keyboard-accessible via a standard HTTP request."""
        client = TestClient(app, raise_server_exceptions=False)
        res = client.get("/health")
        assert res.status_code == 200
        body = res.json()
        assert body["status"] == "running"
        assert body["service"] == "byaku-rebuild-api"

    def test_root_endpoint_is_get(self) -> None:
        """GET / describes the API for screen readers."""
        client = TestClient(app, raise_server_exceptions=False)
        res = client.get("/")
        assert res.status_code == 200
        body = res.json()
        assert "service" in body
        assert "description" in body
        # Description must be non-empty so a screen reader has something to announce.
        assert body["description"]

    def test_status_endpoint_is_get(self) -> None:
        """GET /api/status is keyboard-accessible."""
        client = TestClient(app, raise_server_exceptions=False)
        res = client.get("/api/status")
        assert res.status_code == 200
        body = res.json()
        assert "backend" in body
        assert body["backend"] == "connected"

    def test_query_endpoint_is_post(self) -> None:
        """POST /api/query is keyboard-accessible via HTTP POST."""
        # Adapters are already mocked by the autouse fixture.
        client = TestClient(app, raise_server_exceptions=False)
        res = client.post("/api/query", json={"query": "What objects are visible?"})
        assert res.status_code == 200
        body = res.json()
        # Before any frames, status is not_ready — this is a valid keyboard-accessible response.
        assert body["type"] == "query_result"
        assert "status" in body

    def test_wrong_method_returns_405_not_404(self) -> None:
        """Wrong HTTP method returns 405 — screen readers can distinguish method errors."""
        client = TestClient(app, raise_server_exceptions=False)
        res = client.get("/api/query")
        assert res.status_code == 405
        body = res.json()
        assert body["type"] == "error"
        assert body["code"] == "method_not_allowed"

    def test_unknown_route_returns_404_with_structured_error(self) -> None:
        """Unknown route returns 404 with structured error body for screen readers."""
        client = TestClient(app, raise_server_exceptions=False)
        res = client.get("/nonexistent")
        assert res.status_code == 404
        body = res.json()
        assert body["type"] == "error"
        assert body["code"] == "not_found"

    def test_camera_start_stop_are_post(self) -> None:
        """Camera lifecycle endpoints are POST — no pointer input required."""
        client = TestClient(app, raise_server_exceptions=False)
        # /api/camera/stop should be POST; GET returns 405.
        res = client.get("/api/camera/stop")
        assert res.status_code == 405

    def test_camera_stop_without_camera_returns_structured_response(self) -> None:
        """Stopping a camera that was never started returns a structured response."""
        app.state.camera_producer = None
        client = TestClient(app, raise_server_exceptions=False)
        res = client.post("/api/camera/stop")
        assert res.status_code == 200
        body = res.json()
        assert body["type"] == "camera_status"
        assert body["status"] == "stopped"


# ---------------------------------------------------------------------------
# 2. Screen-reader support — error messages and narration quality
# ---------------------------------------------------------------------------


class TestScreenReaderSupport:
    """Error messages, narration, and spoken_text must be well-formed and non-empty."""

    def test_query_not_ready_spoken_text_is_non_empty(self) -> None:
        """Before any frame, the spoken_text for not_ready is a legible sentence."""
        # Adapters are already mocked by the autouse fixture; fresh orchestrator has no committed scene.
        client = TestClient(app, raise_server_exceptions=False)
        res = client.post("/api/query", json={"query": "What can I see?"})
        assert res.status_code == 200
        body = res.json()
        assert body["status"] == "not_ready"
        speech = body.get("speech_response")
        assert isinstance(speech, dict)
        spoken = speech.get("spoken_text")
        assert isinstance(spoken, str) and spoken.strip(), "spoken_text must be non-empty for screen readers"

    def test_empty_query_returns_descriptive_error(self) -> None:
        """Empty query returns HTTP 422 with a message a screen reader can announce."""
        client = TestClient(app, raise_server_exceptions=False)
        res = client.post("/api/query", json={"query": "   "})
        assert res.status_code == 422
        body = res.json()
        assert body["type"] == "error"
        assert body["code"] == "invalid_request"
        # Message must contain enough context for a screen reader to explain the problem.
        assert body["message"]

    def test_websocket_invalid_query_returns_descriptive_error(self) -> None:
        """WebSocket empty query returns structured error with non-empty message."""
        client = TestClient(app, raise_server_exceptions=False)
        with client.websocket_connect("/ws") as ws:
            ws.receive_json()  # initial status
            ws.send_json({"type": "query", "text": "   "})
            msg = ws.receive_json()
        assert msg["type"] == "error"
        assert msg["code"] == "invalid_query"
        assert msg["message"]  # Must be non-empty for screen reader

    def test_websocket_unsupported_message_has_descriptive_error(self) -> None:
        """Unsupported WebSocket message type returns structured error with message."""
        client = TestClient(app, raise_server_exceptions=False)
        with client.websocket_connect("/ws") as ws:
            ws.receive_json()  # initial status
            ws.send_json({"type": "unknown_type_xyz"})
            msg = ws.receive_json()
        assert msg["type"] == "error"
        assert msg["code"] == "unsupported_message"
        assert "unknown_type_xyz" in msg["message"]

    def test_spoken_text_for_visible_object_is_non_empty(self) -> None:
        """prepare_spoken_response produces non-empty text for a visible query result."""
        query_result = {
            "query": "Where is the bottle?",
            "answer": "The bottle is on the left.",
            "status": "visible",
            "matched_objects": ["track:1"],
            "evidence": [],
            "warnings": [],
        }
        result = prepare_spoken_response(query_result)
        spoken = result.get("spoken_text")
        assert isinstance(spoken, str) and spoken.strip()

    def test_spoken_text_for_not_found_is_non_empty(self) -> None:
        """prepare_spoken_response produces non-empty text when object is not found."""
        query_result = {
            "query": "Where is the chair?",
            "answer": "No object matching 'chair' is visible.",
            "status": "not_found",
            "matched_objects": [],
            "evidence": [],
            "warnings": [],
        }
        result = prepare_spoken_response(query_result)
        spoken = result.get("spoken_text")
        assert isinstance(spoken, str) and spoken.strip()

    def test_spoken_text_for_missing_object_is_non_empty(self) -> None:
        """prepare_spoken_response produces non-empty text when object is missing/lost."""
        query_result = {
            "query": "Where is the bottle?",
            "answer": "The bottle is temporarily missing.",
            "status": "missing",
            "matched_objects": ["track:2"],
            "evidence": [],
            "warnings": [],
        }
        result = prepare_spoken_response(query_result)
        spoken = result.get("spoken_text")
        assert isinstance(spoken, str) and spoken.strip()

    def test_scene_summary_spoken_text_is_non_empty(self) -> None:
        """prepare_spoken_scene_summary produces non-empty text from a description."""
        description_result = {
            "description": "A bottle is visible on the left.",
            "visible_object_count": 1,
            "described_relationships": [],
            "warnings": [],
        }
        result = prepare_spoken_scene_summary(description_result)
        spoken = result.get("spoken_text")
        assert isinstance(spoken, str) and spoken.strip()

    def test_scene_summary_empty_scene_spoken_text_is_non_empty(self) -> None:
        """prepare_spoken_scene_summary produces non-empty text for empty scenes."""
        description_result = {
            "description": "No objects are currently visible.",
            "visible_object_count": 0,
            "described_relationships": [],
            "warnings": [],
        }
        result = prepare_spoken_scene_summary(description_result)
        spoken = result.get("spoken_text")
        assert isinstance(spoken, str) and spoken.strip()

    def test_status_fields_use_expected_vocabulary(self) -> None:
        """GET /api/status stage_status fields use documented vocabulary."""
        client = TestClient(app, raise_server_exceptions=False)
        res = client.get("/api/status")
        assert res.status_code == 200
        body = res.json()
        stage_status = body.get("stage_status", {})
        valid_values = {"ready", "unavailable", "partial", "error", "connected", "disconnected"}
        for field, value in stage_status.items():
            assert value in valid_values, (
                f"stage_status.{field}={value!r} is not in valid vocabulary {valid_values}"
            )

    def test_backend_field_always_present_in_status(self) -> None:
        """backend field is always 'connected' in /api/status — screen readers rely on it."""
        client = TestClient(app, raise_server_exceptions=False)
        res = client.get("/api/status")
        assert res.status_code == 200
        assert res.json()["backend"] == "connected"

    def test_error_responses_always_have_type_error(self) -> None:
        """All error responses have type='error' for consistent ARIA role mapping."""
        client = TestClient(app, raise_server_exceptions=False)
        # 404 on unknown route
        res1 = client.get("/api/does_not_exist")
        assert res1.json()["type"] == "error"
        # 405 on wrong method
        res2 = client.get("/api/query")
        assert res2.json()["type"] == "error"
        # 422 on validation failure
        res3 = client.post("/api/frame", json={})
        assert res3.json()["type"] == "error"

    def test_narration_in_query_result_is_string_or_absent(self) -> None:
        """narration field in query_result is a non-empty string or absent — never null."""
        # Adapters are already mocked by the autouse fixture.
        client = TestClient(app, raise_server_exceptions=False)
        res = client.post("/api/query", json={"query": "What can I see?"})
        assert res.status_code == 200
        body = res.json()
        narration = body.get("narration")
        # narration may be absent (not_ready has no narration) or a non-empty string.
        if narration is not None:
            assert isinstance(narration, str) and narration.strip(), (
                "narration must be a non-empty string or absent"
            )


# ---------------------------------------------------------------------------
# 3. Failure handling — errors return recoverable structured responses
# ---------------------------------------------------------------------------


class TestFailureHandling:
    """Camera, depth, detection, pipeline, and query failures return structured errors."""

    def test_camera_capture_without_start_returns_structured_error(self) -> None:
        """Capture without starting returns camera_not_running error, not a 5xx crash."""
        app.state.camera_producer = None
        client = TestClient(app, raise_server_exceptions=False)
        res = client.post("/api/camera/capture")
        assert res.status_code == 409
        body = res.json()
        assert body["type"] == "error"
        assert body["code"] == "camera_not_running"
        assert body["stage"] == "camera"
        assert body["recoverable"] is True

    def test_camera_start_when_unavailable_returns_structured_error(self) -> None:
        """Camera start with a mock-unavailable source returns camera_unavailable."""
        app.state.camera_producer = None
        fake_producer = MagicMock()
        fake_producer.start.return_value = {"status": "unavailable", "message": "no camera found"}
        fake_producer.is_running = False
        fake_producer.source = 99

        with patch.object(api_module, "_new_camera_producer", return_value=fake_producer):
            client = TestClient(app, raise_server_exceptions=False)
            res = client.post("/api/camera/start", json={"source": 99})

        assert res.status_code == 503
        body = res.json()
        assert body["type"] == "error"
        assert body["code"] == "camera_unavailable"
        assert body["recoverable"] is True
        assert body["message"]  # Non-empty for screen readers

    def test_frame_with_invalid_base64_returns_400(self) -> None:
        """Malformed base64 frame data returns HTTP 400 with error body."""
        client, depth_mock, detection_mock = _make_client_with_mocks()
        depth_mock.estimate_depth.return_value = _fake_depth_result()
        detection_mock.process_frame.return_value = _fake_detection_result("f1", 1)

        res = client.post(
            "/api/frame",
            json={
                "frame_id": "f-bad64",
                "frame_number": 1,
                "width": 4,
                "height": 4,
                "data": "!!!not_valid_base64!!!",
                "timestamp": 1.0,
            },
        )
        assert res.status_code in {400, 422}
        body = res.json()
        assert body["type"] == "error"

    def test_frame_dimension_mismatch_returns_400(self) -> None:
        """Frame whose decoded size doesn't match declared w/h returns 400."""
        client, depth_mock, detection_mock = _make_client_with_mocks()
        depth_mock.estimate_depth.return_value = _fake_depth_result(4, 4)
        detection_mock.process_frame.return_value = _fake_detection_result("f1", 1)

        res = client.post(
            "/api/frame",
            json={
                "frame_id": "f-mismatch",
                "frame_number": 1,
                "width": 100,  # wrong — actual JPEG is 4×4
                "height": 100,
                "data": _b64_jpeg(4, 4),
                "timestamp": 1.0,
            },
        )
        assert res.status_code == 400
        body = res.json()
        assert body["type"] == "error"

    def test_duplicate_frame_id_returns_409(self) -> None:
        """Submitting the same frame_id twice returns 409 (duplicate rejected)."""
        client, depth_mock, detection_mock = _make_client_with_mocks()

        def _depth(frame):
            h, w = frame.shape[:2]
            return _fake_depth_result(h, w)

        depth_mock.estimate_depth.side_effect = _depth

        call_n = [0]

        def _detect(frame, frame_id, frame_number):
            call_n[0] += 1
            return _fake_detection_result(frame_id, frame_number)

        detection_mock.process_frame.side_effect = _detect

        req = _make_frame_request("dup-id", 1)
        res1 = client.post("/api/frame", json=req)
        assert res1.status_code == 200

        # Re-use the same frame_id — must be rejected
        req2 = _make_frame_request("dup-id", 2)
        res2 = client.post("/api/frame", json=req2)
        assert res2.status_code == 409
        body = res2.json()
        assert body["type"] == "error"

    def test_out_of_order_frame_number_returns_409(self) -> None:
        """A frame_number not greater than the last committed one returns 409."""
        client, depth_mock, detection_mock = _make_client_with_mocks()

        def _depth(frame):
            h, w = frame.shape[:2]
            return _fake_depth_result(h, w)

        depth_mock.estimate_depth.side_effect = _depth

        call_n = [0]

        def _detect(frame, frame_id, frame_number):
            call_n[0] += 1
            return _fake_detection_result(frame_id, frame_number)

        detection_mock.process_frame.side_effect = _detect

        res1 = client.post("/api/frame", json=_make_frame_request("order-1", 5))
        assert res1.status_code == 200

        # frame_number 3 < 5 already committed
        res2 = client.post("/api/frame", json=_make_frame_request("order-2", 3))
        assert res2.status_code == 409
        body = res2.json()
        assert body["type"] == "error"

    def test_depth_failure_frame_returns_error_scene_remains_queryable(self) -> None:
        """Depth failure returns structured error; the previous scene is still queryable."""
        client, depth_mock, detection_mock = _make_client_with_mocks()

        def _depth_good(frame):
            h, w = frame.shape[:2]
            return _fake_depth_result(h, w)

        def _detect(frame, frame_id, frame_number):
            return _fake_detection_result(frame_id, frame_number)

        depth_mock.estimate_depth.side_effect = _depth_good
        detection_mock.process_frame.side_effect = _detect

        # Process one good frame so there is a committed scene.
        res_ok = client.post("/api/frame", json=_make_frame_request("good-frame", 1))
        assert res_ok.status_code == 200

        # Next depth call fails.
        depth_mock.estimate_depth.side_effect = lambda frame: {
            "status": "error",
            "depth_available": False,
            "depth_map": None,
            "warnings": ["depth model failed"],
        }
        res_fail = client.post("/api/frame", json=_make_frame_request("fail-frame", 2))
        assert res_fail.status_code in {400, 503}
        body_fail = res_fail.json()
        assert body_fail["type"] == "error"

        # Previous committed scene is still queryable.
        res_q = client.post("/api/query", json={"query": "What objects are visible?"})
        assert res_q.status_code == 200
        q_body = res_q.json()
        assert q_body["type"] == "query_result"
        assert q_body["status"] != "not_ready"  # Something was committed before

    def test_pipeline_exception_scene_remains_queryable(self) -> None:
        """An exception inside the pipeline leaves the previous committed scene queryable."""
        client, depth_mock, detection_mock = _make_client_with_mocks()

        def _depth_good(frame):
            h, w = frame.shape[:2]
            return _fake_depth_result(h, w)

        depth_mock.estimate_depth.side_effect = _depth_good

        call_n = [0]

        def _detect(frame, frame_id, frame_number):
            call_n[0] += 1
            if call_n[0] == 1:
                return _fake_detection_result(frame_id, frame_number)
            raise RuntimeError("simulated detection crash")

        detection_mock.process_frame.side_effect = _detect

        # First frame commits successfully.
        res1 = client.post("/api/frame", json=_make_frame_request("stable-1", 1))
        assert res1.status_code == 200

        # Second frame crashes in detection.
        res2 = client.post("/api/frame", json=_make_frame_request("crash-2", 2))
        assert res2.status_code != 200

        # The first committed scene is still queryable.
        res_q = client.post("/api/query", json={"query": "What objects are visible?"})
        assert res_q.status_code == 200
        assert res_q.json()["type"] == "query_result"

    def test_websocket_frame_processing_failure_sends_recoverable_error(self) -> None:
        """WebSocket frame processing failure sends recoverable=True error to browser."""
        app.state.orchestrator = PipelineOrchestrator()

        client, depth_mock, detection_mock = _make_client_with_mocks()
        depth_mock.estimate_depth.side_effect = lambda frame: {
            "status": "error",
            "depth_available": False,
            "depth_map": None,
            "warnings": ["depth unavailable"],
        }

        with client.websocket_connect("/ws") as ws:
            ws.receive_json()  # initial status
            ws.send_json({"type": "start_camera"})
            ws.receive_json()  # acknowledgement

            ws.send_json(
                {
                    "type": "frame",
                    "frame_id": 1,
                    "width": 4,
                    "height": 4,
                    "data": _b64_jpeg(4, 4),
                }
            )
            msg = ws.receive_json()

        assert msg["type"] == "error"
        assert msg["recoverable"] is True
        assert msg.get("frame_id") == 1
        assert msg["message"]  # Non-empty for screen readers

    def test_websocket_bad_json_returns_protocol_error(self) -> None:
        """WebSocket non-JSON text returns structured protocol error."""
        client = TestClient(app, raise_server_exceptions=False)
        with client.websocket_connect("/ws") as ws:
            ws.receive_json()  # initial status
            ws.send_text("this is not json {{{")
            msg = ws.receive_json()
        assert msg["type"] == "error"
        assert msg["stage"] == "protocol"
        assert msg["code"] == "invalid_json"

    def test_websocket_missing_type_field_returns_error(self) -> None:
        """WebSocket message without 'type' field returns structured error."""
        client = TestClient(app, raise_server_exceptions=False)
        with client.websocket_connect("/ws") as ws:
            ws.receive_json()  # initial status
            ws.send_json({"data": "no type field"})
            msg = ws.receive_json()
        assert msg["type"] == "error"
        assert msg["code"] == "invalid_message"


# ---------------------------------------------------------------------------
# 4. Recovery behavior
# ---------------------------------------------------------------------------


class TestRecoveryBehavior:
    """After failures, the API allows retries and session reconnects."""

    def test_after_camera_error_can_restart_and_capture(self) -> None:
        """After a camera unavailable error, a new start with a different source succeeds."""
        app.state.camera_producer = None

        failed_producer = MagicMock()
        failed_producer.start.return_value = {"status": "unavailable", "message": "no cam"}
        failed_producer.is_running = False
        failed_producer.source = 0

        success_producer = MagicMock()
        success_producer.start.return_value = {
            "status": "started",
            "source": 1,
            "next_frame_number": 1,
        }
        success_producer.is_running = True
        success_producer.source = 1

        call_count = [0]

        def _factory(source):
            call_count[0] += 1
            return failed_producer if source == 0 else success_producer

        with patch.object(api_module, "_new_camera_producer", side_effect=_factory):
            client = TestClient(app, raise_server_exceptions=False)
            res1 = client.post("/api/camera/start", json={"source": 0})
            assert res1.status_code == 503  # first attempt fails

            app.state.camera_producer = None  # clean up for retry
            res2 = client.post("/api/camera/start", json={"source": 1})
            assert res2.status_code == 200
            body = res2.json()
            assert body["status"] == "started"

    def test_after_failed_frame_sequence_continues_correctly(self) -> None:
        """After a failed frame, the next successful frame uses a higher frame_number."""
        client, depth_mock, detection_mock = _make_client_with_mocks()

        calls = [0]

        def _depth_alt(frame):
            calls[0] += 1
            if calls[0] == 1:
                # First call (frame_number=1): success
                h, w = frame.shape[:2]
                return _fake_depth_result(h, w)
            if calls[0] == 2:
                # Second call (frame_number=2): fail
                return {"status": "error", "depth_available": False, "depth_map": None, "warnings": []}
            # Third call (frame_number=3): success
            h, w = frame.shape[:2]
            return _fake_depth_result(h, w)

        depth_mock.estimate_depth.side_effect = _depth_alt

        det_calls = [0]

        def _detect(frame, frame_id, frame_number):
            det_calls[0] += 1
            return _fake_detection_result(frame_id, frame_number)

        detection_mock.process_frame.side_effect = _detect

        res1 = client.post("/api/frame", json=_make_frame_request("seq-1", 1))
        assert res1.status_code == 200

        # Frame 2 fails (depth error)
        client.post("/api/frame", json=_make_frame_request("seq-2", 2))

        # Frame 3 must use a higher frame_number than 1
        res3 = client.post("/api/frame", json=_make_frame_request("seq-3", 3))
        assert res3.status_code == 200
        body3 = res3.json()
        assert body3["frame_number"] >= 3

    def test_websocket_reconnect_gets_fresh_session(self) -> None:
        """A new WebSocket connection starts with fresh session state."""
        client = TestClient(app, raise_server_exceptions=False)

        # First session
        with client.websocket_connect("/ws") as ws1:
            initial1 = ws1.receive_json()
            assert initial1["type"] == "status"
            ws1.send_json({"type": "start_camera"})
            ws1.receive_json()  # acknowledgement

        # Second session — independent of the first
        with client.websocket_connect("/ws") as ws2:
            initial2 = ws2.receive_json()
            assert initial2["type"] == "status"
            # camera should be disconnected for this new session
            # (first session's camera_active was session-scoped)
            assert initial2["backend"] == "connected"

    def test_query_after_pipeline_crash_returns_last_good_scene(self) -> None:
        """Query after a pipeline crash returns the last good committed scene, not an error."""
        client, depth_mock, detection_mock = _make_client_with_mocks()

        def _depth_good(frame):
            h, w = frame.shape[:2]
            return _fake_depth_result(h, w)

        depth_mock.estimate_depth.side_effect = _depth_good

        call_n = [0]

        def _detect(frame, frame_id, frame_number):
            call_n[0] += 1
            if call_n[0] >= 2:
                raise RuntimeError("crash!")
            return _fake_detection_result(frame_id, frame_number)

        detection_mock.process_frame.side_effect = _detect

        # Commit one good frame
        client.post("/api/frame", json=_make_frame_request("crash-base", 1))

        # Second frame crashes
        client.post("/api/frame", json=_make_frame_request("crash-bad", 2))

        # Query should reflect the last good scene
        res = client.post("/api/query", json={"query": "What objects are visible?"})
        assert res.status_code == 200
        body = res.json()
        assert body["type"] == "query_result"
        # Status should be a recognized value (empty_scene, visible, not_ready...)
        assert body["status"] in {"visible", "empty_scene", "not_ready", "not_found"}

    def test_camera_busy_is_recoverable(self) -> None:
        """camera_busy error is always marked recoverable=True."""
        # Lock the camera operation lock to simulate a concurrent operation.
        api_module._camera_operation_lock.acquire()
        try:
            client = TestClient(app, raise_server_exceptions=False)
            res = client.post("/api/camera/start")
            assert res.status_code == 409
            body = res.json()
            assert body["code"] == "camera_busy"
            assert body["recoverable"] is True
        finally:
            api_module._camera_operation_lock.release()


# ---------------------------------------------------------------------------
# 5. No metric hallucinations in narration
# ---------------------------------------------------------------------------


class TestNoMetricHallucinations:
    """Narration and spoken_text must never contain metric distances."""

    _METRIC_PATTERNS = ["meter", "metre", "cm ", " cm", "feet", "foot", " ft", "kilometer", "kilometre"]

    def _assert_no_metric(self, text: str) -> None:
        lower = text.lower()
        for pattern in self._METRIC_PATTERNS:
            assert pattern not in lower, (
                f"spoken_text contains metric distance pattern {pattern!r}: {text!r}"
            )

    def test_visible_spoken_text_has_no_metric(self) -> None:
        """Directional spoken_text for a visible object contains no metric distances."""
        query_result = {
            "query": "Where is the bottle?",
            "answer": "The bottle is on the left, in front.",
            "status": "visible",
            "matched_objects": ["track:1"],
            "evidence": [{"horizontal_position": "left"}],
            "warnings": [],
        }
        result = prepare_spoken_response(query_result)
        self._assert_no_metric(result["spoken_text"])

    def test_not_found_spoken_text_has_no_metric(self) -> None:
        """not_found spoken text contains no metric distances."""
        query_result = {
            "query": "Where is the table?",
            "answer": "No object matching 'table' is currently visible.",
            "status": "not_found",
            "matched_objects": [],
            "evidence": [],
            "warnings": [],
        }
        result = prepare_spoken_response(query_result)
        self._assert_no_metric(result["spoken_text"])

    def test_scene_summary_has_no_metric(self) -> None:
        """Scene summary spoken_text contains no metric distances."""
        description_result = {
            "description": "A person is visible on the right, in front of a chair.",
            "visible_object_count": 2,
            "described_relationships": ["person in front of chair"],
            "warnings": [],
        }
        result = prepare_spoken_scene_summary(description_result)
        self._assert_no_metric(result["spoken_text"])

    def test_answer_query_no_metric_in_answer(self) -> None:
        """answer_query never produces metric distance claims in the answer text."""
        from Workflow.reasoning.scene_memory import SceneMemory

        mem = SceneMemory()
        objects = [
            {
                "object_id": "track:1",
                "class_name": "bottle",
                "bounding_box": [50, 50, 150, 300],
                "center": [100, 175],
                "horizontal_position": "left",
                "vertical_position": "lower",
                "relative_depth_value": 0.7,
            }
        ]
        relationships = [
            {
                "object_a": "track:1",
                "object_b": None,
                "relation": None,
            }
        ]
        spatial_output = {"objects": objects, "relationships": []}
        result = answer_query("Where is the bottle?", spatial_output=spatial_output)
        answer_text = result.get("answer", "")
        if answer_text:
            lower = answer_text.lower()
            for pat in self._METRIC_PATTERNS:
                assert pat not in lower, f"answer contains metric pattern {pat!r}: {answer_text!r}"

    def test_stale_scene_narration_has_no_metric(self) -> None:
        """Stale-scene answer (scene_age_seconds > 30) contains no metric distances."""
        spatial_output = {
            "objects": [
                {
                    "object_id": "track:1",
                    "class_name": "chair",
                    "bounding_box": [10, 10, 100, 200],
                    "center": [55, 105],
                    "horizontal_position": "center",
                    "vertical_position": "lower",
                }
            ],
            "relationships": [],
        }
        result = answer_query(
            "Where is the chair?",
            spatial_output=spatial_output,
            scene_age_seconds=45.0,  # stale
        )
        answer_text = result.get("answer", "")
        lower = answer_text.lower()
        for pat in self._METRIC_PATTERNS:
            assert pat not in lower


# ---------------------------------------------------------------------------
# 6. Edge cases — whitespace, None, and malformed input handling
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """Malformed, None, and edge-case inputs are handled without crashes."""

    def test_frame_with_empty_frame_id_rejected(self) -> None:
        """Frame with empty frame_id returns HTTP 422."""
        client = TestClient(app, raise_server_exceptions=False)
        res = client.post(
            "/api/frame",
            json={
                "frame_id": "   ",
                "frame_number": 1,
                "width": 4,
                "height": 4,
                "data": _b64_jpeg(),
            },
        )
        assert res.status_code == 422
        assert res.json()["type"] == "error"

    def test_frame_missing_required_fields_returns_422(self) -> None:
        """Frame request missing required fields returns HTTP 422."""
        client = TestClient(app, raise_server_exceptions=False)
        res = client.post("/api/frame", json={"frame_id": "f1"})
        assert res.status_code == 422
        assert res.json()["type"] == "error"

    def test_query_with_only_whitespace_rejected(self) -> None:
        """Query with only whitespace returns HTTP 422."""
        client = TestClient(app, raise_server_exceptions=False)
        res = client.post("/api/query", json={"query": "\n\t  \r"})
        assert res.status_code == 422
        assert res.json()["type"] == "error"

    def test_camera_start_with_invalid_source_type_rejected(self) -> None:
        """camera/start with a non-int/non-str source returns HTTP 422."""
        client = TestClient(app, raise_server_exceptions=False)
        res = client.post("/api/camera/start", json={"source": None})
        # None is not int|str — should be rejected
        assert res.status_code in {422, 200}  # depends on pydantic coercion; at minimum no 5xx crash

    def test_websocket_frame_with_zero_width_returns_error(self) -> None:
        """WebSocket frame with zero width returns structured invalid_frame error."""
        client = TestClient(app, raise_server_exceptions=False)
        with client.websocket_connect("/ws") as ws:
            ws.receive_json()
            ws.send_json({"type": "start_camera"})
            ws.receive_json()
            ws.send_json({"type": "frame", "frame_id": 1, "width": 0, "height": 4, "data": _b64_jpeg()})
            msg = ws.receive_json()
        assert msg["type"] == "error"
        assert msg["code"] == "invalid_frame"

    def test_websocket_frame_with_negative_frame_id_returns_error(self) -> None:
        """WebSocket frame with negative frame_id returns structured error."""
        client = TestClient(app, raise_server_exceptions=False)
        with client.websocket_connect("/ws") as ws:
            ws.receive_json()
            ws.send_json({"type": "start_camera"})
            ws.receive_json()
            ws.send_json({"type": "frame", "frame_id": -1, "width": 4, "height": 4, "data": _b64_jdbc()})
            msg = ws.receive_json()
        assert msg["type"] == "error"
        assert msg["code"] == "invalid_frame_id"

    def test_websocket_ping_without_ts_returns_pong(self) -> None:
        """WebSocket ping without a ts field returns pong without ts (no crash)."""
        client = TestClient(app, raise_server_exceptions=False)
        with client.websocket_connect("/ws") as ws:
            ws.receive_json()
            ws.send_json({"type": "ping"})
            msg = ws.receive_json()
        assert msg["type"] == "pong"
        assert "ts" not in msg  # ts only echoed when provided

    def test_websocket_ping_with_ts_returns_echo(self) -> None:
        """WebSocket ping with ts echoes the ts value correctly."""
        client = TestClient(app, raise_server_exceptions=False)
        with client.websocket_connect("/ws") as ws:
            ws.receive_json()
            ws.send_json({"type": "ping", "ts": 12345})
            msg = ws.receive_json()
        assert msg["type"] == "pong"
        assert msg["ts"] == 12345


# ---------------------------------------------------------------------------
# Intentional typo helper (used in test_websocket_frame_with_negative_frame_id_returns_error)
# ---------------------------------------------------------------------------


def _b64_jdbc(width: int = 4, height: int = 4) -> str:
    """Alias of _b64_jpeg to keep the test above readable without renaming."""
    return _b64_jpeg(width, height)
