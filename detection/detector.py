from __future__ import annotations

from typing import Any, Iterable, List

import numpy as np
from ultralytics import YOLO


class YOLODetector:
    """Simple YOLO-based detector for a single OpenCV frame."""

    def __init__(self, model_name: str = "yolo11n.pt", confidence_threshold: float = 0.5) -> None:
        self.model_name = model_name
        self.confidence_threshold = float(confidence_threshold)
        self.model = YOLO(model_name)
        self.model_names = getattr(self.model, "names", {}) or {}

    def _class_name_from_value(self, value: Any) -> str:
        if isinstance(value, str):
            return value.lower()

        if isinstance(value, (int, float)):
            index = int(value)
            if isinstance(self.model_names, dict):
                label = self.model_names.get(index, "object")
                return str(label).lower()
            if isinstance(self.model_names, list):
                return str(self.model_names[index]).lower() if index < len(self.model_names) else "object"
            return "object"

        return "object"

    @staticmethod
    def _coerce_number_list(value: Any) -> list[float]:
        if value is None:
            return []
        if isinstance(value, (list, tuple, np.ndarray)):
            return [float(item) for item in value]
        return []

    def _parse_detection_items(self, result: Any) -> Iterable[tuple[float, float, float, float, float, Any]]:
        boxes = getattr(result, "boxes", None)
        if boxes is None:
            return []

        if hasattr(boxes, "xyxy") and hasattr(boxes, "conf") and hasattr(boxes, "cls"):
            xyxy = boxes.xyxy
            conf = boxes.conf
            cls = boxes.cls
            if isinstance(xyxy, np.ndarray) and isinstance(conf, np.ndarray) and isinstance(cls, np.ndarray):
                for i in range(len(conf)):
                    yield (*[float(v) for v in xyxy[i]], float(conf[i]), cls[i])
                return

        raw = list(boxes)
        for item in raw:
            if not isinstance(item, (list, tuple)) or len(item) < 3:
                continue
            bbox = item[0]
            confidence = float(item[1])
            class_value = item[2]
            coords = self._coerce_number_list(bbox)
            if len(coords) != 4:
                continue
            yield (coords[0], coords[1], coords[2], coords[3], confidence, class_value)

    def detect(self, frame: Any) -> List[dict]:
        """Run YOLO on a single frame and return filtered detections."""
        if frame is None or not isinstance(frame, np.ndarray):
            return []

        if frame.size == 0:
            return []

        try:
            raw_results = self.model(frame, conf=self.confidence_threshold, verbose=False)
        except Exception:
            return []

        if not isinstance(raw_results, (list, tuple)):
            return []

        detections: List[dict] = []
        for result in raw_results:
            for x1, y1, x2, y2, confidence, class_value in self._parse_detection_items(result):
                if confidence < self.confidence_threshold:
                    continue

                detections.append(
                    {
                        "class_name": self._class_name_from_value(class_value),
                        "confidence": float(confidence),
                        "bounding_box": [float(x1), float(y1), float(x2), float(y2)],
                    }
                )

        return detections
