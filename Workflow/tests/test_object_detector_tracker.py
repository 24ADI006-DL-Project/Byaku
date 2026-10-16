from pathlib import Path
import os
import subprocess
import sys

import cv2
import numpy as np
import pytest

from Workflow.vision.object_detector_tracker import DEFAULT_WEIGHTS, DetectionTrackingAdapter
from Workflow.scene_pipeline import PipelineOrchestrator


class FakeBoxes:
    def __init__(self, xyxy, confidence, classes, track_ids=None):
        self.xyxy = np.asarray(xyxy, dtype=np.float32).reshape((-1, 4))
        self.conf = np.asarray(confidence, dtype=np.float32).reshape((-1,))
        self.cls = np.asarray(classes, dtype=np.float32).reshape((-1,))
        self.id = None if track_ids is None else np.asarray(track_ids, dtype=np.float32).reshape((-1,))

    def __len__(self):
        return len(self.conf)


class FakeResult:
    names = {0: "person", 1: "bottle"}

    def __init__(self, boxes):
        self.boxes = boxes


class FakeModel:
    names = FakeResult.names

    def __init__(self, results_by_frame=None, error=None):
        self.results_by_frame = list(results_by_frame or [])
        self.error = error
        self.calls = []

    def track(self, frame, **kwargs):
        self.calls.append((frame.copy(), kwargs))
        if self.error is not None:
            raise self.error
        if self.results_by_frame:
            return [self.results_by_frame.pop(0)]
        return [FakeResult(FakeBoxes([], [], [], []))]


def fake_result(boxes=None, confidences=None, classes=None, track_ids=None):
    return FakeResult(FakeBoxes(boxes or [], confidences or [], classes or [], track_ids))


def test_model_initialization_uses_existing_local_weights() -> None:
    assert DEFAULT_WEIGHTS.is_file()
    adapter = DetectionTrackingAdapter()

    assert Path(adapter.weights_path) == DEFAULT_WEIGHTS.resolve()
    assert adapter.model is not None


def test_ultralytics_settings_persist_across_process_restarts_without_reset(tmp_path) -> None:
    environment = os.environ.copy()
    environment["YOLO_CONFIG_DIR"] = str(tmp_path)
    script = (
        "from ultralytics import settings; "
        f"settings.update({{'datasets_dir': {str(tmp_path / 'datasets')!r}, "
        f"'weights_dir': {str(tmp_path / 'Models')!r}, "
        f"'runs_dir': {str(tmp_path / 'Models' / 'training_runs')!r}}}); "
        "print(settings.file)"
    )
    first = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        check=True,
        env=environment,
    )
    second = subprocess.run(
        [sys.executable, "-c", "from ultralytics import settings; print(settings.file); print(settings['runs_dir'])"],
        capture_output=True,
        text=True,
        check=True,
        env=environment,
    )

    assert Path(second.stdout.splitlines()[0]).is_file()
    assert str(tmp_path / "Models" / "training_runs") in second.stdout
    assert "reset to default values" not in first.stderr + second.stderr


def test_project_ultralytics_settings_load_from_the_runtime_config() -> None:
    project_settings = Path(__file__).resolve().parents[2] / "Config" / "ultralytics" / "Ultralytics" / "settings.json"
    script = (
        "from Workflow.vision.object_detector_tracker import _configure_ultralytics_settings; "
        "_configure_ultralytics_settings(); "
        "from ultralytics import settings; "
        "print(settings.file); "
        "print(settings['datasets_dir']); "
        "print(settings['weights_dir']); "
        "print(settings['runs_dir'])"
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        check=True,
        cwd=Path(__file__).resolve().parents[2],
    )
    lines = result.stdout.splitlines()

    assert Path(lines[0]).resolve() == project_settings.resolve()
    assert lines[1].endswith(r"Data\datasets")
    assert lines[2].endswith(r"Models")
    assert lines[3].endswith(r"Models\training_runs")
    assert "reset to default values" not in result.stderr


def test_detection_schema_preserves_class_confidence_box_and_thresholds() -> None:
    model = FakeModel(
        [fake_result([[5, 6, 25, 36]], [0.88], [1], [42])]
    )
    adapter = DetectionTrackingAdapter(
        model=model,
        confidence_threshold=0.31,
        iou_threshold=0.62,
    )
    frame = np.zeros((48, 64, 3), dtype=np.uint8)

    result = adapter.process_frame(frame, frame_id="unit-1", frame_number=1)

    assert result["status"] == "success"
    assert result["model"] == str(DEFAULT_WEIGHTS.resolve())
    assert result["frame"] == {
        "frame_id": "unit-1",
        "frame_number": 1,
        "image_shape": (48, 64),
        "width": 64,
        "height": 48,
    }
    assert result["detections"] == [
        {"class_name": "bottle", "confidence": pytest.approx(0.88), "bounding_box": [5.0, 6.0, 25.0, 36.0]}
    ]
    assert result["tracks"][0]["track_id"] == 42
    assert result["orchestrator_detections"] == [
        {"track_id": 42, "class_name": "bottle", "confidence": pytest.approx(0.88), "bounding_box": [5.0, 6.0, 25.0, 36.0]}
    ]
    _, kwargs = model.calls[0]
    assert kwargs == {
        "conf": 0.31,
        "iou": 0.62,
        "imgsz": 640,
        "persist": True,
        "tracker": "bytetrack.yaml",
        "verbose": False,
    }


def test_fake_tracker_id_remains_stable_across_frames() -> None:
    model = FakeModel(
        [
            fake_result([[5, 6, 25, 36]], [0.88], [0], [11]),
            fake_result([[6, 6, 26, 36]], [0.86], [0], [11]),
        ]
    )
    adapter = DetectionTrackingAdapter(model=model)
    frame = np.zeros((48, 64, 3), dtype=np.uint8)

    first = adapter.process_frame(frame, frame_id="frame-1", frame_number=1)
    second = adapter.process_frame(frame, frame_id="frame-2", frame_number=2)

    assert first["tracks"][0]["track_id"] == second["tracks"][0]["track_id"] == 11
    assert all(call[1]["persist"] is True for call in model.calls)


def test_missing_track_ids_are_not_invented_and_empty_detections_are_valid() -> None:
    model = FakeModel(
        [
            fake_result([[5, 6, 25, 36]], [0.72], [0], None),
            fake_result(),
        ]
    )
    adapter = DetectionTrackingAdapter(model=model)
    frame = np.zeros((48, 64, 3), dtype=np.uint8)

    untracked = adapter.process_frame(frame, frame_id="frame-1", frame_number=1)
    empty = adapter.process_frame(frame, frame_id="frame-2", frame_number=2)

    assert len(untracked["detections"]) == 1
    assert untracked["tracks"] == []
    assert "track_id" not in untracked["orchestrator_detections"][0]
    assert untracked["untracked_detection_count"] == 1
    assert any("identities were not invented" in warning for warning in untracked["warnings"])
    assert empty["status"] == "success"
    assert empty["empty_detections"] is True
    assert empty["detections"] == []
    assert empty["orchestrator_detections"] == []


def test_invalid_frame_reuse_and_model_errors_return_structured_failures() -> None:
    model = FakeModel([fake_result()])
    adapter = DetectionTrackingAdapter(model=model)
    frame = np.zeros((48, 64, 3), dtype=np.uint8)

    invalid = adapter.process_frame(None, frame_id="bad", frame_number=1)
    accepted = adapter.process_frame(frame, frame_id="frame-1", frame_number=1)
    repeated = adapter.process_frame(frame, frame_id="frame-1", frame_number=2)
    non_increasing = adapter.process_frame(frame, frame_id="frame-2", frame_number=1)

    assert invalid["status"] == "invalid_input"
    assert accepted["status"] == "success"
    assert repeated["status"] == "invalid_input"
    assert non_increasing["status"] == "invalid_input"
    assert len(model.calls) == 1

    failing = DetectionTrackingAdapter(model=FakeModel(error=RuntimeError("synthetic model failure")))
    failure = failing.process_frame(frame, frame_id="error-1", frame_number=1)
    assert failure["status"] == "error"
    assert "synthetic model failure" in failure["warnings"][0]


def test_model_loading_errors_and_missing_weights_are_reported() -> None:
    with pytest.raises(FileNotFoundError, match="YOLO weights not found"):
        DetectionTrackingAdapter(weights_path=Path("missing-yolo-weights.pt"))

    def fail_to_load(_path: str):
        raise OSError("synthetic load error")

    with pytest.raises(RuntimeError, match="synthetic load error"):
        DetectionTrackingAdapter(model_factory=fail_to_load)


@pytest.mark.parametrize(
    ("confidence", "iou"),
    [(-0.1, 0.5), (1.1, 0.5), (0.25, -0.1), (0.25, 1.1)],
)
def test_invalid_thresholds_are_rejected(confidence: float, iou: float) -> None:
    with pytest.raises(ValueError):
        DetectionTrackingAdapter(
            model=FakeModel(),
            confidence_threshold=confidence,
            iou_threshold=iou,
        )


def test_real_local_yolo_bytetrack_results_flow_into_orchestrator() -> None:
    workspace_root = Path(__file__).resolve().parents[2]
    image_path = workspace_root / "Data" / "samples" / "bedroom_scene_desk.jpg"
    assert image_path.is_file()
    frame = cv2.imread(str(image_path))
    assert frame is not None

    adapter = DetectionTrackingAdapter()
    tracked = adapter.process_frame(frame, frame_id="real-test1", frame_number=1)

    assert tracked["model"] == str(DEFAULT_WEIGHTS.resolve())
    assert tracked["status"] in {"success", "partial"}
    assert len(tracked["detections"]) > 0
    height, width = frame.shape[:2]
    for item in tracked["detections"]:
        assert isinstance(item["class_name"], str) and item["class_name"]
        assert 0 <= item["confidence"] <= 1
        x1, y1, x2, y2 = item["bounding_box"]
        assert 0 <= x1 < x2 <= width
        assert 0 <= y1 < y2 <= height

    pipeline = PipelineOrchestrator()
    pipeline_result = pipeline.process_frame(
        frame_id=tracked["frame"]["frame_id"],
        frame_number=tracked["frame"]["frame_number"],
        image_dimensions=tracked["frame"]["image_shape"],
        detections=tracked["orchestrator_detections"],
    )

    assert pipeline_result["status"] in {"success", "partial"}
    assert len(pipeline_result["objects"]) == len(tracked["orchestrator_detections"])
    assert pipeline_result["frame"]["frame_id"] == tracked["frame"]["frame_id"]
    assert all(item["class_name"] in {detection["class_name"] for detection in tracked["detections"]}
               for item in pipeline_result["objects"])