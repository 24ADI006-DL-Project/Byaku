import base64

import cv2
import numpy as np
import pytest

import Workflow.scene_pipeline as pipeline_module
from Backend.fastapi_application import app
from Workflow.scene_pipeline import PipelineOrchestrator
from Backend.live_camera_producer import LiveCameraProducer


class FakeCapture:
    def __init__(self, reads, *, opened=True):
        self.reads = list(reads)
        self.opened = opened
        self.released = False
        self.read_calls = 0

    def isOpened(self):
        return self.opened

    def read(self):
        self.read_calls += 1
        if not self.reads:
            return False, None
        result = self.reads.pop(0)
        if isinstance(result, Exception):
            raise result
        return result

    def release(self):
        self.released = True


def make_frame(value=0):
    return np.full((12, 18, 3), value, dtype=np.uint8)


def successful_submitter(request):
    return {
        "type": "frame_result",
        "frame_id": request.frame_id,
        "frame_number": request.frame_number,
        "status": "success",
    }


def make_producer(capture, submitted, *, source="mock-camera", **kwargs):
    return LiveCameraProducer(
        source,
        capture_factory=lambda configured_source: submitted.append(configured_source) or capture,
        frame_submitter=kwargs.pop("frame_submitter", successful_submitter),
        frame_number_provider=kwargs.pop("frame_number_provider", lambda: 0),
        **kwargs,
    )


def test_successful_start_capture_and_submission_use_frame_api_contract():
    capture = FakeCapture([(True, make_frame(10)), (True, make_frame(20))])
    opened_sources = []
    submitted_requests = []
    producer = make_producer(
        capture,
        opened_sources,
        frame_submitter=lambda request: submitted_requests.append(request) or successful_submitter(request),
    )

    started = producer.start()
    first = producer.capture_once()
    second = producer.capture_once()

    assert started["status"] == "started"
    assert opened_sources == ["mock-camera"]
    assert first["status"] == second["status"] == "submitted"
    assert [request.frame_number for request in submitted_requests] == [1, 2]
    assert submitted_requests[0].frame_id != submitted_requests[1].frame_id
    for request in submitted_requests:
        assert request.width == 18 and request.height == 12
        decoded = cv2.imdecode(
            np.frombuffer(base64.b64decode(request.data), dtype=np.uint8),
            cv2.IMREAD_COLOR,
        )
        assert decoded.shape == (12, 18, 3)
    assert first["frame_result"]["status"] == "success"
    producer.stop()
    assert capture.released is True


def test_unavailable_camera_is_reported_and_capture_is_released():
    capture = FakeCapture([], opened=False)
    producer = make_producer(capture, [])

    result = producer.start()

    assert result["status"] == "unavailable"
    assert producer.is_running is False
    assert capture.released is True
    assert producer.capture_once()["status"] == "not_running"


def test_camera_factory_exception_is_reported_without_leaving_producer_running():
    def fail_open(_source):
        raise OSError("camera open exploded")

    producer = LiveCameraProducer(
        "unavailable-camera",
        capture_factory=fail_open,
        frame_submitter=successful_submitter,
        frame_number_provider=lambda: 0,
    )

    result = producer.start()

    assert result["status"] == "camera_error"
    assert "camera open exploded" in result["message"]
    assert producer.is_running is False


def test_read_failure_can_recover_then_releases_after_failure_limit():
    capture = FakeCapture([(False, None), (True, make_frame()), (False, None), (False, None)])
    submitted = []
    producer = make_producer(capture, [], frame_submitter=lambda request: submitted.append(request) or successful_submitter(request), max_consecutive_read_failures=2)
    producer.start()

    first_read = producer.capture_once()
    recovered = producer.capture_once()
    failed_again = producer.capture_once()
    terminated = producer.capture_once()

    assert first_read["status"] == "read_failed" and first_read["recoverable"] is True
    assert recovered["status"] == "submitted"
    assert failed_again["recoverable"] is True
    assert terminated["recoverable"] is False
    assert producer.is_running is False
    assert len(submitted) == 1
    assert capture.released is True


def test_repeated_start_stop_and_restart_are_idempotent_and_keep_sequence():
    captures = [FakeCapture([(True, make_frame())]), FakeCapture([(True, make_frame())])]
    opened_sources = []
    submitted_requests = []

    def capture_factory(source):
        opened_sources.append(source)
        return captures[len(opened_sources) - 1]

    producer = LiveCameraProducer(
        "camera-A",
        capture_factory=capture_factory,
        frame_submitter=lambda request: submitted_requests.append(request) or successful_submitter(request),
        frame_number_provider=lambda: 0,
    )

    assert producer.start()["status"] == "started"
    assert producer.start()["status"] == "already_running"
    assert producer.capture_once()["frame_number"] == 1
    assert producer.stop()["already_stopped"] is False
    assert producer.stop()["already_stopped"] is True
    assert producer.start()["status"] == "started"
    assert producer.capture_once()["frame_number"] == 2
    producer.stop()

    assert opened_sources == ["camera-A", "camera-A"]
    assert captures[0].released and captures[1].released
    assert submitted_requests[0].frame_id != submitted_requests[1].frame_id


def test_capture_exception_and_submitter_exception_both_release_camera():
    failed_capture = FakeCapture([RuntimeError("capture exploded")])
    capture_producer = make_producer(failed_capture, [])
    capture_producer.start()
    capture_result = capture_producer.capture_once()
    assert capture_result["status"] == "capture_error"
    assert failed_capture.released is True

    failed_submit_capture = FakeCapture([(True, make_frame())])

    def fail_submit(_request):
        raise RuntimeError("submission exploded")

    submit_producer = make_producer(failed_submit_capture, [], frame_submitter=fail_submit)
    submit_producer.start()
    submit_result = submit_producer.capture_once()
    assert submit_result["status"] == "submission_failed"
    assert "submission exploded" in submit_result["message"]
    assert failed_submit_capture.released is True


def test_non_successful_api_response_is_not_reported_as_submitted():
    capture = FakeCapture([(True, make_frame())])
    producer = make_producer(
        capture,
        [],
        frame_submitter=lambda _request: {"type": "error", "status": "error"},
    )
    producer.start()

    result = producer.capture_once()

    assert result["status"] == "submission_failed"
    assert result["status"] != "submitted"
    assert capture.released is True


def test_run_releases_after_limit_and_does_not_mask_submission_failure():
    limited_capture = FakeCapture([(True, make_frame()), (True, make_frame())])
    limited_producer = make_producer(limited_capture, [])
    limited = limited_producer.run(max_frames=1)
    assert limited["status"] == "completed"
    assert limited["frames_submitted"] == 1
    assert limited_capture.released is True

    failed_capture = FakeCapture([(True, make_frame())])

    def fail_submit(_request):
        raise RuntimeError("submission failure")

    failed_producer = make_producer(failed_capture, [], frame_submitter=fail_submit)
    failed = failed_producer.run(max_frames=2)
    assert failed["status"] == "submission_failed"
    assert failed["frames_submitted"] == 0
    assert failed_capture.released is True


class FakeDetectionAdapter:
    def __init__(self):
        self._seen_frame_ids = set()
        self._last_frame_number = None

    def process_frame(self, frame, *, frame_id, frame_number):
        self._seen_frame_ids.add(frame_id)
        self._last_frame_number = frame_number
        detection = {
            "track_id": 8,
            "class_name": "bottle",
            "confidence": 0.9,
            "bounding_box": [1.0, 1.0, 8.0, 8.0],
        }
        return {
            "status": "success",
            "orchestrator_detections": [detection],
            "warnings": [],
        }


class FakeDepthAdapter:
    def estimate_depth(self, frame):
        height, width = frame.shape[:2]
        return {
            "status": "success",
            "depth_available": True,
            "depth_map": np.linspace(0.0, 1.0, height * width, dtype=np.float32).reshape(height, width),
            "warnings": [],
        }


def test_failed_api_submission_keeps_previous_scene_queryable(monkeypatch):
    app.state.orchestrator = PipelineOrchestrator()
    app.state.detection_adapter = FakeDetectionAdapter()
    app.state.depth_adapter = FakeDepthAdapter()
    capture = FakeCapture([(True, make_frame()), (True, make_frame(1))])
    producer = LiveCameraProducer(
        0,
        capture_factory=lambda _source: capture,
        frame_number_provider=lambda: app.state.orchestrator._latest_frame_number or 0,
    )
    producer.start()
    first = producer.capture_once()
    assert first["status"] == "submitted"
    previous_query = app.state.orchestrator.query("Is the bottle currently visible?")

    def fail_description(*args, **kwargs):
        raise RuntimeError("synthetic scene-description failure")

    monkeypatch.setattr(pipeline_module, "describe_scene", fail_description)
    failed = producer.capture_once()

    assert failed["status"] == "submission_failed"
    assert producer.is_running is False
    assert capture.released is True
    assert app.state.orchestrator._latest_frame_number == first["frame_number"]
    assert app.state.orchestrator._latest_frame_id == first["frame_id"]
    assert app.state.orchestrator.query("Is the bottle currently visible?") == previous_query
    assert app.state.detection_adapter._last_frame_number == first["frame_number"]
    assert len(app.state.detection_adapter._seen_frame_ids) == 1


@pytest.mark.parametrize("max_frames", [0, -1, True])
def test_run_rejects_invalid_frame_limits(max_frames):
    producer = LiveCameraProducer(frame_submitter=successful_submitter, frame_number_provider=lambda: 0)

    result = producer.run(max_frames=max_frames)

    assert result["status"] == "invalid_input"