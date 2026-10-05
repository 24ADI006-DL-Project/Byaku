import numpy as np

import detection.detector as detector_module


class FakeYOLOResult:
    def __init__(self, boxes):
        self.boxes = boxes


class FakeYOLO:
    names = {0: "person", 1: "chair", 2: "bottle"}

    def __init__(self, model_name):
        self.model_name = model_name
        self.names = self.__class__.names

    def __call__(self, frame, conf=None, verbose=False):
        if getattr(frame, "shape", None) is None:
            raise TypeError("frame must be a valid NumPy array")

        if frame.shape[:2] == (64, 64):
            return [
                FakeYOLOResult([
                    ([[10, 20, 100, 200], 0.92, 0]),
                    ([[200, 50, 300, 300], 0.35, 1]),
                ])
            ]
        return [FakeYOLOResult([])]


def test_detector_initializes_model(monkeypatch):
    monkeypatch.setattr(detector_module, "YOLO", FakeYOLO)

    d = detector_module.YOLODetector(model_name="yolo11n.pt", confidence_threshold=0.5)

    assert d.model_name == "yolo11n.pt"
    assert d.confidence_threshold == 0.5
    assert d.model is not None


def test_detector_returns_valid_detection_structure(monkeypatch):
    monkeypatch.setattr(detector_module, "YOLO", FakeYOLO)

    d = detector_module.YOLODetector(confidence_threshold=0.5)
    frame = np.zeros((64, 64, 3), dtype=np.uint8)

    detections = d.detect(frame)

    assert isinstance(detections, list)
    assert len(detections) == 1
    item = detections[0]
    assert item["class_name"] == "person"
    assert 0 <= item["confidence"] <= 1
    assert isinstance(item["bounding_box"], list)
    assert len(item["bounding_box"]) == 4


def test_detector_uses_confidence_threshold(monkeypatch):
    class ThresholdYOLO(FakeYOLO):
        def __call__(self, frame, conf=None, verbose=False):
            return [
                FakeYOLOResult([([10, 20, 100, 200], 0.91, 2)]),
                FakeYOLOResult([([200, 50, 300, 300], 0.35, 1)]),
            ]

    monkeypatch.setattr(detector_module, "YOLO", ThresholdYOLO)
    d = detector_module.YOLODetector(confidence_threshold=0.5)

    detections = d.detect(np.zeros((64, 64, 3), dtype=np.uint8))

    assert len(detections) == 1
    assert detections[0]["class_name"] == "bottle"


def test_detector_handles_no_detections(monkeypatch):
    class EmptyYOLO(FakeYOLO):
        def __call__(self, frame, conf=None, verbose=False):
            return [FakeYOLOResult([])]

    monkeypatch.setattr(detector_module, "YOLO", EmptyYOLO)
    d = detector_module.YOLODetector(confidence_threshold=0.5)

    assert d.detect(np.zeros((64, 64, 3), dtype=np.uint8)) == []


def test_detector_handles_invalid_input(monkeypatch):
    monkeypatch.setattr(detector_module, "YOLO", FakeYOLO)
    d = detector_module.YOLODetector(confidence_threshold=0.5)

    assert d.detect(None) == []
