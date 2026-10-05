from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest
import torch

from byaku_rebuild.depth_integration import (
    DEPTH_INPUT_SIZE,
    MODEL_NAME,
    DepthAnythingV2Adapter,
    normalize_relative_depth_map,
)
from byaku_rebuild.detection_tracking import DEFAULT_WEIGHTS, DetectionTrackingAdapter
from byaku_rebuild.frame_profiler import FrameProfiler
from byaku_rebuild.pipeline_orchestrator import PipelineOrchestrator


WORKSPACE_ROOT = Path(__file__).resolve().parents[2]


class FakeProcessor:
    def __init__(self):
        self.size = None
        self.keep_aspect_ratio = True
        self.ensure_multiple_of = 14
        self.do_resize = True
        self.do_pad = False
        self.crop_size = None

    def __call__(self, images, return_tensors, *, size=None):
        self.size = size
        return {"pixel_values": torch.zeros((1, 3, 14, 28), dtype=torch.float32)}


class FakeDepthModel:
    def __init__(self, values=None, error=None, missing_output=False):
        self.values = values
        self.error = error
        self.missing_output = missing_output
        self.device = None
        self.evaluated = False

    def to(self, device):
        self.device = device
        return self

    def eval(self):
        self.evaluated = True
        return self

    def __call__(self, **inputs):
        if self.error is not None:
            raise self.error
        if self.missing_output:
            return SimpleNamespace()
        return SimpleNamespace(predicted_depth=torch.as_tensor(self.values, dtype=torch.float32))


class FakeFactory:
    def __init__(self, value=None, error=None):
        self.value = value
        self.error = error
        self.calls = []

    def from_pretrained(self, model_name, **kwargs):
        self.calls.append((model_name, kwargs))
        if self.error is not None:
            raise self.error
        return self.value


@pytest.fixture(scope="module")
def real_depth_adapter() -> DepthAnythingV2Adapter:
    return DepthAnythingV2Adapter(device="cpu")


@pytest.mark.parametrize("image_name", ["test1.jpg", "test2.jpg"])
def test_real_cached_model_inference_returns_valid_frame_sized_relative_depth(
    real_depth_adapter: DepthAnythingV2Adapter,
    image_name: str,
) -> None:
    image_path = WORKSPACE_ROOT / image_name
    image = cv2.imread(str(image_path))
    assert image is not None
    height, width = image.shape[:2]

    result = real_depth_adapter.estimate_depth(image)

    assert result["status"] == "success"
    assert result["depth_available"] is True
    assert result["metadata"]["model"] == MODEL_NAME
    assert result["metadata"]["input_shape"] == (height, width)
    assert result["depth_map"].shape == (height, width)
    assert np.isfinite(result["depth_map"]).all()
    assert np.min(result["depth_map"]) >= 0
    assert np.max(result["depth_map"]) <= 1
    assert np.ptp(result["depth_map"]) > 0
    assert result["metadata"]["depth_ordering"] == "higher_is_nearer"
    assert result["metadata"]["metric_distance_available"] is False
    assert result["inference_time_ms"] > 0

    upper_background = result["depth_map"][
        int(height * 0.12):int(height * 0.35),
        int(width * 0.35):int(width * 0.68),
    ]
    lower_foreground = result["depth_map"][
        int(height * 0.78):int(height * 0.95),
        int(width * 0.35):int(width * 0.68),
    ]
    assert np.median(lower_foreground) > np.median(upper_background)


def test_model_initialization_is_local_only_and_uses_the_configured_model() -> None:
    processor_factory = FakeFactory(value=FakeProcessor())
    model_factory = FakeFactory(value=FakeDepthModel(np.arange(4, dtype=np.float32).reshape(1, 2, 2)))

    adapter = DepthAnythingV2Adapter(
        processor_factory=processor_factory,
        model_factory=model_factory,
        device="cpu",
    )

    assert adapter.model_name == MODEL_NAME
    assert processor_factory.calls[0] == (MODEL_NAME, {"local_files_only": True})
    assert model_factory.calls[0] == (MODEL_NAME, {"local_files_only": True})
    assert adapter.model.evaluated is True
    assert DEFAULT_WEIGHTS.is_file()


def test_depth_processor_uses_reduced_compatible_input_size() -> None:
    processor = FakeProcessor()
    adapter = DepthAnythingV2Adapter(
        processor=processor,
        model=FakeDepthModel(np.array([[[0.0, 1.0], [2.0, 3.0]]], dtype=np.float32)),
        device="cpu",
    )

    profiler = FrameProfiler()
    result = adapter.estimate_depth(np.zeros((8, 12, 3), dtype=np.uint8), profiler=profiler)

    assert result["status"] == "success"
    assert processor.size == {"height": DEPTH_INPUT_SIZE, "width": DEPTH_INPUT_SIZE}
    assert DEPTH_INPUT_SIZE == 364
    assert profiler.summary()["metadata"] == {
        "depth_processor_source_image_shape": [8, 12, 3],
        "depth_processor_size_argument": {"height": 364, "width": 364},
        "depth_processor_keep_aspect_ratio": True,
        "depth_processor_ensure_multiple_of": 14,
        "depth_processor_do_resize": True,
        "depth_processor_do_pad": False,
        "depth_processor_crop_size": None,
        "depth_processor_class": "FakeProcessor",
        "depth_processor_tensor_shape": [1, 3, 14, 28],
        "depth_processor_input_size": 364,
        "depth_model_input_tensor_shape": [1, 3, 14, 28],
        "depth_model_output_tensor_shape": [1, 2, 2],
    }


def test_model_loading_failure_is_reported_without_retry_or_download() -> None:
    processor_factory = FakeFactory(value=FakeProcessor())
    model_factory = FakeFactory(error=OSError("cached model files unavailable"))

    with pytest.raises(RuntimeError, match="downloads disabled.*cached model files unavailable"):
        DepthAnythingV2Adapter(
            processor_factory=processor_factory,
            model_factory=model_factory,
            device="cpu",
        )

    assert model_factory.calls == [(MODEL_NAME, {"local_files_only": True})]


def test_normalizer_resizes_and_preserves_relative_ordering() -> None:
    raw_depth = np.array([[0.0, 1.0], [2.0, 3.0]], dtype=np.float32)

    result = normalize_relative_depth_map(raw_depth, (8, 12))

    assert result["status"] == "success"
    assert result["depth_map"].shape == (8, 12)
    assert np.isfinite(result["depth_map"]).all()
    assert result["depth_map"][0, 0] < result["depth_map"][-1, -1]
    assert result["metadata"]["depth_ordering"] == "higher_is_nearer"


def test_constant_or_invalid_depth_maps_are_unavailable_not_fabricated() -> None:
    constant = normalize_relative_depth_map(np.full((4, 5), 2.0, dtype=np.float32), (4, 5))
    non_finite = normalize_relative_depth_map(np.array([[0.0, np.nan], [1.0, 2.0]]), (2, 2))
    wrong_rank = normalize_relative_depth_map(np.zeros((2, 2, 2), dtype=np.float32), (2, 2))

    assert constant["status"] == "partial"
    assert constant["depth_map"] is None
    assert constant["depth_available"] is False
    assert any("constant" in warning for warning in constant["warnings"])
    assert non_finite["status"] == "error"
    assert non_finite["depth_map"] is None
    assert any("NaN or infinite" in warning for warning in non_finite["warnings"])
    assert wrong_rank["status"] == "error"


def test_relative_depth_samples_create_depth_relationships() -> None:
    raw_depth = np.tile(np.linspace(1.0, 0.0, 40, dtype=np.float32), (20, 1))
    normalized = normalize_relative_depth_map(raw_depth, (20, 40))
    pipeline = PipelineOrchestrator()
    result = pipeline.process_frame(
        frame_id="relative-test",
        frame_number=1,
        image_dimensions=(20, 40),
        detections=[
            {"track_id": 1, "class_name": "bottle", "confidence": 0.9, "bounding_box": (0, 0, 10, 10)},
            {"track_id": 2, "class_name": "chair", "confidence": 0.8, "bounding_box": (30, 0, 40, 10)},
        ],
        depth_map=normalized["depth_map"],
    )

    assert result["status"] == "success"
    assert result["objects"][0]["relative_depth_value"] > result["objects"][1]["relative_depth_value"]
    assert any(
        relation["relation"] == "nearer_than" and relation["subject"] == "track:1"
        for relation in result["relationships"]
    )


def test_inference_errors_missing_outputs_and_invalid_frames_are_structured() -> None:
    frame = np.zeros((8, 12, 3), dtype=np.uint8)
    depth_error = DepthAnythingV2Adapter(
        processor=FakeProcessor(),
        model=FakeDepthModel(error=RuntimeError("synthetic inference error")),
        device="cpu",
    ).estimate_depth(frame)
    missing_output = DepthAnythingV2Adapter(
        processor=FakeProcessor(),
        model=FakeDepthModel(missing_output=True),
        device="cpu",
    ).estimate_depth(frame)
    invalid_frame = DepthAnythingV2Adapter(
        processor=FakeProcessor(),
        model=FakeDepthModel(np.ones((1, 2, 2), dtype=np.float32)),
        device="cpu",
    ).estimate_depth(np.zeros((4, 4), dtype=np.uint8))

    assert depth_error["status"] == "error"
    assert "synthetic inference error" in depth_error["warnings"][0]
    assert missing_output["status"] == "error"
    assert "predicted_depth" in missing_output["warnings"][0]
    assert invalid_frame["status"] == "invalid_input"


def test_depth_adapter_preserves_detection_and_tracking_identity() -> None:
    frame = np.zeros((8, 12, 3), dtype=np.uint8)
    depth_adapter = DepthAnythingV2Adapter(
        processor=FakeProcessor(),
        model=FakeDepthModel(np.array([[[0.0, 1.0], [2.0, 3.0]]], dtype=np.float32)),
        device="cpu",
    )
    orchestrator = PipelineOrchestrator()
    detection_result = {
        "status": "success",
        "frame": {"frame_id": "tracked-frame", "frame_number": 1, "image_shape": (8, 12)},
        "orchestrator_detections": [
            {"track_id": 71, "class_name": "bottle", "confidence": 0.93, "bounding_box": [1, 1, 6, 7]}
        ],
    }

    result = depth_adapter.process_frame(
        frame,
        detection_result=detection_result,
        orchestrator=orchestrator,
    )

    assert result["status"] == "success"
    assert result["frame"] == detection_result["frame"]
    object_record = result["pipeline_result"]["objects"][0]
    assert object_record["object_id"] == "track:71"
    assert object_record["track_id"] == 71
    assert object_record["confidence"] == 0.93
    assert result["depth"]["depth_map"].shape == (8, 12)


def test_empty_detections_and_depth_failure_still_use_pipeline_without_depth() -> None:
    frame = np.zeros((8, 12, 3), dtype=np.uint8)
    depth_adapter = DepthAnythingV2Adapter(
        processor=FakeProcessor(),
        model=FakeDepthModel(error=RuntimeError("depth unavailable")),
        device="cpu",
    )
    pipeline = PipelineOrchestrator()
    detection_result = {
        "status": "success",
        "frame": {"frame_id": "empty-frame", "frame_number": 1, "image_shape": (8, 12)},
        "orchestrator_detections": [],
    }

    result = depth_adapter.process_frame(frame, detection_result=detection_result, orchestrator=pipeline)

    assert result["status"] == "partial"
    assert result["depth"]["depth_available"] is False
    assert result["pipeline_result"]["objects"] == []
    assert result["pipeline_result"]["spatial"]["objects"] == []
    assert any("depth unavailable" in warning for warning in result["warnings"])


def test_real_yolo_bytetrack_and_depth_flow_into_orchestrator() -> None:
    image_path = WORKSPACE_ROOT / "test1.jpg"
    image = cv2.imread(str(image_path))
    assert image is not None
    detection_adapter = DetectionTrackingAdapter()
    detection_result = detection_adapter.process_frame(image, frame_id="depth-e2e-test1", frame_number=1)
    assert detection_result["status"] in {"success", "partial"}
    assert detection_result["detections"]

    depth_adapter = DepthAnythingV2Adapter(device="cpu")
    orchestrator = PipelineOrchestrator()
    result = depth_adapter.process_frame(
        image,
        detection_result=detection_result,
        orchestrator=orchestrator,
    )

    assert result["status"] in {"success", "partial"}
    assert result["depth"]["depth_available"] is True
    assert result["depth"]["depth_map"].shape == image.shape[:2]
    assert np.isfinite(result["depth"]["depth_map"]).all()
    input_tracks = detection_result["orchestrator_detections"]
    output_objects = result["pipeline_result"]["objects"]
    assert len(output_objects) == len(input_tracks)
    for source, output in zip(input_tracks, output_objects):
        assert output["object_id"] == f"track:{source['track_id']}" if source.get("track_id") is not None else output["object_id"].startswith("temporary:")
        assert output["class_name"] == source["class_name"]
        assert output.get("confidence") == source.get("confidence")
        assert output["relative_depth_value"] is not None