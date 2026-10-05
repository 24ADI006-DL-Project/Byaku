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
        return [
            FakeYOLOResult(
                [
                    ([10, 20, 100, 200], 0.92, 0),
                    ([200, 50, 300, 300], 0.35, 1),
                ]
            )
        ]


detector_module.YOLO = FakeYOLO
d = detector_module.YOLODetector(confidence_threshold=0.5)
print(d.detect(np.zeros((64, 64, 3), dtype=np.uint8)))
