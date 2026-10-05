from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from app import AppConfig, VisionAssistantApp
class FakeDetector:
    def __init__(self, detections: list[dict[str, Any]] | None = None) -> None:
        self.detections = detections or []
        self.calls = 0

    def detect(self, frame: np.ndarray) -> list[dict[str, Any]]:
        self.calls += 1
        return list(self.detections)


class FakeTracker:
    def __init__(self) -> None:
        self.reset_called = False

    def update(self, detections: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return [
            {
                "track_id": index + 1,
                "class_name": item["class_name"],
                "confidence": item["confidence"],
                "bounding_box": item["bounding_box"],
            }
            for index, item in enumerate(detections)
        ]

    def reset(self) -> None:
        self.reset_called = True


class FakeDepthEstimator:
    def estimate_depth(self, frame: np.ndarray) -> np.ndarray:
        return np.full(frame.shape[:2], 0.3, dtype=np.float32)


class FakeEngine:
    def __init__(self) -> None:
        self.spoken: list[str] = []

    def say(self, text: str) -> None:
        self.spoken.append(text)

    def runAndWait(self) -> None:
        return None

    def save_to_file(self, text: str, path: str) -> None:
        Path(path).write_bytes(text.encode("utf-8"))


class FakeTTS:
    def __init__(self) -> None:
        self.engine = FakeEngine()
        self.spoken: list[str] = []

    def speak(self, text: str) -> bool:
        self.spoken.append(text)
        return True


class FakeSpeechInput:
    last_status = "ready"
    last_error = None

    def listen(self) -> str | None:
        return None


def make_app(
    detections: list[dict[str, Any]] | None = None,
    **components: Any,
) -> tuple[VisionAssistantApp, FakeDetector, FakeTracker, FakeTTS]:
    detector = components.pop("detector", FakeDetector(detections))
    tracker = components.pop("tracker", FakeTracker())
    tts = components.pop("tts", FakeTTS())
    application = VisionAssistantApp(
        config=AppConfig(processing_fps=5, frame_skip=0),
        detector=detector,
        tracker=tracker,
        depth_estimator=FakeDepthEstimator(),
        tts=tts,
        speech_input=FakeSpeechInput(),
        **components,
    )
    return application, detector, tracker, tts


def test_application_initialization_and_module_wiring() -> None:
    app, detector, tracker, _ = make_app()

    assert app.detector is detector
    assert app.tracker is tracker
    assert app.query_engine.scene is app.scene_memory
    assert app.scene_memory.scene_graph is not None
    assert app.config.processing_fps == 5


def test_frame_processing_runs_detection_tracking_depth_and_scene_memory() -> None:
    app, detector, _, _ = make_app(
        [{"class_name": "bottle", "confidence": 0.9, "bounding_box": [5, 5, 30, 35]}]
    )

    result = app.process_frame(np.zeros((48, 64, 3), dtype=np.uint8))

    assert detector.calls == 1
    assert len(result["detections"]) == 1
    assert len(result["tracks"]) == len(result["objects"]) == 1
    assert np.isclose(result["objects"][0]["relative_depth"], 0.3)
    assert result["objects"][0]["horizontal_position"] == "left"
    assert result["frame"].shape == (48, 64, 3)
    assert result["scene"]["objects"] == result["objects"]


def test_query_flows_through_scene_graph_to_narration_and_tts_audio() -> None:
    app, _, _, _ = make_app(
        [{"class_name": "bottle", "confidence": 0.9, "bounding_box": [5, 5, 30, 35]}]
    )
    app.process_frame(np.zeros((48, 64, 3), dtype=np.uint8))

    result = app.answer_query("Where is my bottle?")

    assert result["query_result"]["status"] == "found"
    assert result["narration"] == "Your bottle is close and on your left."
    assert result["audio"] is not None
    assert Path(result["audio"]).is_file()


def test_event_narration_is_sent_to_tts_without_blocking_frame_processing() -> None:
    app, _, _, tts = make_app(
        [{"class_name": "person", "confidence": 0.95, "bounding_box": [5, 5, 30, 35]}]
    )

    result = app.process_frame(np.zeros((48, 64, 3), dtype=np.uint8))
    thread = app._speech_thread
    if thread is not None:
        thread.join(timeout=1)

    assert result["events"][0]["message_required"] is True
    assert result["events"][0]["narration"] == "Your person is close and on your left."
    assert tts.spoken == [result["events"][0]["narration"]]


def test_empty_detection_case_updates_tracker_and_returns_empty_scene() -> None:
    app, _, _, _ = make_app()

    result = app.process_frame(np.zeros((48, 64, 3), dtype=np.uint8))

    assert result["detections"] == []
    assert result["tracks"] == []
    assert result["objects"] == []
    assert result["status"] == "Ready"


def test_camera_video_failure_is_reported_without_raising() -> None:
    app, _, _, _ = make_app()

    result = app.process_video("missing_video_file.mp4")

    assert result["video"] is None
    assert result["status"] == "Video file could not be opened"


def test_invalid_frame_is_handled_gracefully() -> None:
    app, _, _, _ = make_app()

    assert app.process_frame(None)["status"] == "No frame available"


def test_graceful_shutdown_clears_state_and_stops_processing() -> None:
    app, _, tracker, _ = make_app(
        [{"class_name": "chair", "confidence": 0.9, "bounding_box": [5, 5, 30, 35]}]
    )
    app.process_frame(np.zeros((48, 64, 3), dtype=np.uint8))
    assert app.scene_memory.objects

    app.close()

    assert tracker.reset_called is True
    assert app.scene_memory.objects == []
    assert app.process_frame(np.zeros((48, 64, 3), dtype=np.uint8))["status"] == "Application is shut down"
    app.close()


def test_config_rejects_nonpositive_processing_fps() -> None:
    try:
        AppConfig(processing_fps=0)
    except ValueError as exc:
        assert "processing_fps" in str(exc)
    else:
        raise AssertionError("invalid processing FPS should be rejected")
