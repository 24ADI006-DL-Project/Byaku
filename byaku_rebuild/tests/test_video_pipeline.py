from pathlib import Path

import cv2
import numpy as np
import pytest

import byaku_rebuild.video_pipeline as video_pipeline_module
from byaku_rebuild.video_pipeline import OfflineVideoPipeline


WORKSPACE_ROOT = Path(__file__).resolve().parents[2]


class FakeCapture:
    def __init__(self, frames, *, opened=True, fps=10.0):
        self.frames = list(frames)
        self.opened = opened
        self.fps = fps
        self.index = 0
        self.released = False

    def isOpened(self):
        return self.opened

    def get(self, property_id):
        if property_id == cv2.CAP_PROP_FPS:
            return self.fps
        if property_id == cv2.CAP_PROP_FRAME_COUNT:
            return len(self.frames)
        if property_id == cv2.CAP_PROP_FRAME_WIDTH:
            return self.frames[0].shape[1] if self.frames else 24
        if property_id == cv2.CAP_PROP_FRAME_HEIGHT:
            return self.frames[0].shape[0] if self.frames else 16
        if property_id == cv2.CAP_PROP_POS_MSEC:
            return self.index * 100.0
        return 0.0

    def read(self):
        if not self.opened or self.index >= len(self.frames):
            return False, None
        frame = self.frames[self.index]
        self.index += 1
        return True, frame.copy()

    def release(self):
        self.released = True


class FakeDetectionAdapter:
    def __init__(self, *, fail=False):
        self.fail = fail
        self.calls = []

    def process_frame(self, frame, *, frame_id, frame_number):
        self.calls.append((frame_id, frame_number, int(frame[0, 0, 0])))
        if self.fail:
            raise RuntimeError("synthetic detector error")
        detection = {
            "track_id": 77,
            "class_name": "bottle",
            "confidence": 0.91,
            "bounding_box": [1.0, 1.0, 8.0, 8.0],
        }
        return {
            "status": "success",
            "frame": {
                "frame_id": frame_id,
                "frame_number": frame_number,
                "image_shape": frame.shape[:2],
                "width": frame.shape[1],
                "height": frame.shape[0],
            },
            "detections": [{key: value for key, value in detection.items() if key != "track_id"}],
            "tracks": [detection],
            "orchestrator_detections": [detection],
            "untracked_detection_count": 0,
            "warnings": [],
        }


class FakeDepthAdapter:
    model_name = "synthetic-depth-test-double"

    def __init__(self, *, fail=False, invalid_map=False):
        self.fail = fail
        self.invalid_map = invalid_map
        self.calls = 0

    def estimate_depth(self, frame):
        self.calls += 1
        if self.fail:
            return {
                "status": "error",
                "depth_available": False,
                "depth_map": None,
                "metadata": {},
                "inference_time_ms": 2.0,
                "warnings": ["synthetic depth error"],
            }
        shape = frame.shape[:2]
        depth = np.ones((shape[0] - 1, shape[1]), dtype=np.float32) if self.invalid_map else np.linspace(
            0.0, 1.0, shape[0] * shape[1], dtype=np.float32
        ).reshape(shape)
        return {
            "status": "success",
            "depth_available": True,
            "depth_map": depth,
            "metadata": {"output_shape": tuple(depth.shape), "depth_ordering": "higher_is_nearer"},
            "inference_time_ms": 2.0,
            "warnings": [],
        }


def make_pipeline(capture, *, detector=None, depth=None):
    return OfflineVideoPipeline(
        detection_adapter=detector or FakeDetectionAdapter(),
        depth_adapter=depth or FakeDepthAdapter(),
        capture_factory=lambda _path: capture,
    )


def make_video_path(tmp_path: Path) -> Path:
    path = tmp_path / "synthetic.mp4"
    path.write_bytes(b"synthetic test placeholder")
    return path


def test_missing_and_unreadable_video_paths_return_errors_and_release_capture(tmp_path: Path) -> None:
    pipeline = make_pipeline(FakeCapture([]))
    missing = pipeline.process_video(tmp_path / "missing.mp4")
    assert missing["final_status"] == "error"
    assert "does not exist" in missing["warnings"][0]

    path = make_video_path(tmp_path)
    capture = FakeCapture([], opened=False)
    unreadable = make_pipeline(capture).process_video(path)
    assert unreadable["final_status"] == "error"
    assert "could not open video" in unreadable["warnings"][0]
    assert capture.released is True


def test_empty_video_is_reported_and_capture_is_released(tmp_path: Path) -> None:
    path = make_video_path(tmp_path)
    capture = FakeCapture([])

    result = make_pipeline(capture).process_video(path)

    assert result["final_status"] == "error"
    assert result["frames_read"] == 0
    assert result["frames_processed"] == 0
    assert any("yielded no frames" in warning for warning in result["warnings"])
    assert capture.released is True


def test_frame_limit_sampling_order_metadata_and_timings(tmp_path: Path) -> None:
    path = make_video_path(tmp_path)
    frames = [np.full((16, 24, 3), value, dtype=np.uint8) for value in range(5)]
    capture = FakeCapture(frames, fps=10.0)
    detector = FakeDetectionAdapter()
    pipeline = make_pipeline(capture, detector=detector)

    result = pipeline.process_video(path, max_frames=2, frame_stride=2, query="Where is the bottle?")

    assert result["frames_read"] == 3
    assert result["frames_processed"] == 2
    assert result["frames_skipped"] == 1
    assert result["frames_failed"] == 0
    assert [item["frame"]["frame_number"] for item in result["per_frame_results"]] == [1, 3]
    assert [item["frame"]["timestamp_seconds"] for item in result["per_frame_results"]] == pytest.approx([0.1, 0.3])
    assert [call[2] for call in detector.calls] == [0, 2]
    assert result["per_frame_results"][0]["frame"]["frame_id"] == "synthetic:1"
    assert result["video"]["width"] == 24 and result["video"]["height"] == 16
    assert result["final_status"] == "success"
    assert result["query_result"]["frame_number"] == 3
    assert result["speech_response"]["source_status"] == result["query_result"]["status"]
    assert result["total_processing_time_ms"] >= 0
    assert result["average_processing_time_per_processed_frame_ms"] == pytest.approx(
        result["total_processing_time_ms"] / 2
    )
    assert result["stage_timings_ms"]["detection_tracking"]["status"] == "measured"
    assert result["stage_timings_ms"]["depth_estimation"]["status"] == "measured"
    assert result["stage_timings_ms"]["orchestrator_combined"]["status"] == "measured"
    for stage in ("spatial_processing", "graph_memory_updates", "scene_description"):
        assert result["stage_timings_ms"][stage]["status"] == "not_isolated"
        assert result["stage_timings_ms"][stage]["total_ms"] is None
    assert capture.released is True


def test_sequential_frames_preserve_tracker_and_memory_state(tmp_path: Path) -> None:
    path = make_video_path(tmp_path)
    frames = [np.zeros((16, 24, 3), dtype=np.uint8) for _ in range(3)]
    detector = FakeDetectionAdapter()
    result = make_pipeline(FakeCapture(frames), detector=detector).process_video(path)

    assert result["frames_processed"] == 3
    assert [call[1] for call in detector.calls] == [1, 2, 3]
    assert all(item["track_count"] == 1 for item in result["per_frame_results"])
    assert all(item["objects"][0]["object_id"] == "track:77" for item in result["per_frame_results"])
    assert result["per_frame_results"][-1]["memory"]["visible_ids"] == ["track:77"]


def test_depth_failure_is_partial_and_depth_shape_mismatch_is_not_passed(tmp_path: Path) -> None:
    path = make_video_path(tmp_path)
    frame = np.zeros((16, 24, 3), dtype=np.uint8)
    failed_depth = make_pipeline(FakeCapture([frame]), depth=FakeDepthAdapter(fail=True)).process_video(path)

    assert failed_depth["frames_processed"] == 1
    assert failed_depth["final_status"] == "partial"
    assert failed_depth["per_frame_results"][0]["depth"]["depth_available"] is False
    assert failed_depth["per_frame_results"][0]["objects"][0]["relative_depth_value"] is None
    assert any("synthetic depth error" in warning for warning in failed_depth["warnings"])

    invalid_depth = make_pipeline(FakeCapture([frame]), depth=FakeDepthAdapter(invalid_map=True)).process_video(path)
    assert invalid_depth["frames_processed"] == 1
    assert invalid_depth["frames_failed"] == 0
    assert invalid_depth["final_status"] == "partial"
    assert any("invalid map" in warning for warning in invalid_depth["warnings"])
    assert invalid_depth["per_frame_results"][0]["depth"]["depth_available"] is False


def test_capture_resource_is_released_when_detection_raises(tmp_path: Path) -> None:
    path = make_video_path(tmp_path)
    capture = FakeCapture([np.zeros((16, 24, 3), dtype=np.uint8)])
    result = make_pipeline(capture, detector=FakeDetectionAdapter(fail=True)).process_video(path)

    assert capture.released is True
    assert result["frames_failed"] == 1
    assert result["final_status"] == "error"
    assert any("synthetic detector error" in warning for warning in result["warnings"])


def test_model_initialization_errors_are_reported_without_processing(tmp_path: Path, monkeypatch) -> None:
    path = make_video_path(tmp_path)

    def fail_detection(**_kwargs):
        raise RuntimeError("synthetic YOLO initialization error")

    def fail_depth(**_kwargs):
        raise RuntimeError("synthetic depth initialization error")

    monkeypatch.setattr(video_pipeline_module, "DetectionTrackingAdapter", fail_detection)
    monkeypatch.setattr(video_pipeline_module, "DepthAnythingV2Adapter", fail_depth)
    pipeline = OfflineVideoPipeline()
    result = pipeline.process_video(path)

    assert result["final_status"] == "error"
    assert result["frames_read"] == 0
    assert any("YOLO initialization error" in warning for warning in result["warnings"])
    assert any("depth initialization error" in warning for warning in result["warnings"])


def test_invalid_frame_sampling_options_are_rejected(tmp_path: Path) -> None:
    path = make_video_path(tmp_path)
    pipeline = make_pipeline(FakeCapture([]))

    result = pipeline.process_video(path, max_frames=0)
    assert result["final_status"] == "invalid_input"
    result = pipeline.process_video(path, frame_stride=0)
    assert result["final_status"] == "invalid_input"


def test_real_local_video_runs_both_cached_models_through_orchestrator() -> None:
    video_path = WORKSPACE_ROOT / "pinterest-video.mp4"
    assert video_path.is_file()
    pipeline = OfflineVideoPipeline()
    assert pipeline.detection_adapter is not None
    assert pipeline.depth_adapter is not None
    assert Path(pipeline.detection_adapter.weights_path).is_file()
    assert pipeline.depth_adapter.model_name == "depth-anything/Depth-Anything-V2-Small-hf"

    result = pipeline.process_video(
        video_path,
        max_frames=2,
        frame_stride=1,
        query="What objects are visible?",
    )

    assert result["video"]["readable"] is True
    assert result["frames_read"] == 2
    assert result["frames_processed"] == 2
    assert result["frames_failed"] == 0
    assert [item["frame"]["frame_number"] for item in result["per_frame_results"]] == [1, 2]
    assert all(item["depth"]["depth_available"] for item in result["per_frame_results"])
    assert all(item["stage_timings_ms"]["detection_tracking"] >= 0 for item in result["per_frame_results"])
    assert all(item["stage_timings_ms"]["depth_estimation"] >= 0 for item in result["per_frame_results"])
    assert result["final_status"] in {"success", "partial"}
    assert result["query_result"] is not None
    assert result["speech_response"] is not None