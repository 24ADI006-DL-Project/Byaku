from pathlib import Path

import yaml

from Workflow.vision import relative_depth_estimator as depth_integration
from Workflow.vision.object_detector_tracker import (
    DEFAULT_WEIGHTS,
    DetectionTrackingAdapter,
)


ROOT = Path(__file__).resolve().parents[1]


def test_production_detector_and_depth_configuration():
    adapter = DetectionTrackingAdapter(
        model=object(),
        weights_path=DEFAULT_WEIGHTS,
    )

    assert DEFAULT_WEIGHTS.name == "yolo11n.pt"
    assert DEFAULT_WEIGHTS == (
        ROOT.parent / "Models" / "yolo11n.pt"
    ).resolve()
    assert adapter.confidence_threshold == 0.25
    assert adapter.iou_threshold == 0.45
    assert adapter.tracker_config == "bytetrack.yaml"
    assert depth_integration.DEPTH_INPUT_SIZE == 364

    data = yaml.safe_load((ROOT.parent / "Data" / "datasets" / "byaku_detection" / "data.yaml").read_text(encoding="utf-8"))
    assert list(data["names"].values()) == [
        "person", "bottle", "book", "chair", "table",
        "mobile_phone", "laptop", "bag",
    ]
