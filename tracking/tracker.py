from __future__ import annotations

from types import SimpleNamespace
from typing import Any, Sequence

import numpy as np


class _DetectionBatch:
    """NumPy-backed subset of the Ultralytics detection-results interface."""

    def __init__(self, boxes: np.ndarray, confidences: np.ndarray, classes: np.ndarray) -> None:
        self.xyxy = boxes
        self.conf = confidences
        self.cls = classes

        widths_heights = boxes[:, 2:4] - boxes[:, 0:2]
        centers = (boxes[:, 0:2] + boxes[:, 2:4]) / 2
        self.xywh = np.concatenate((centers, widths_heights), axis=1)

    def __len__(self) -> int:
        return len(self.conf)

    def __getitem__(self, indices: Any) -> _DetectionBatch:
        return _DetectionBatch(self.xyxy[indices], self.conf[indices], self.cls[indices])


class ObjectTracker:
    """Assign persistent IDs to detections using Ultralytics ByteTrack."""

    def __init__(self, track_buffer: int = 30) -> None:
        if track_buffer < 1:
            raise ValueError("track_buffer must be at least 1")

        from ultralytics.trackers.byte_tracker import BYTETracker

        self._tracker = BYTETracker(
            SimpleNamespace(
                track_high_thresh=0.25,
                track_low_thresh=0.1,
                new_track_thresh=0.25,
                track_buffer=int(track_buffer),
                match_thresh=0.8,
                fuse_score=True,
            )
        )
        self._class_ids: dict[str, int] = {}
        self._class_names: dict[int, str] = {}

    def _batch_from_detections(self, detections: Sequence[dict]) -> _DetectionBatch:
        boxes: list[list[float]] = []
        confidences: list[float] = []
        classes: list[int] = []

        for detection in detections:
            try:
                class_name = str(detection["class_name"]).strip().lower()
                confidence = float(detection["confidence"])
                box = [float(value) for value in detection["bounding_box"]]
            except (KeyError, TypeError, ValueError):
                continue

            if not class_name or len(box) != 4 or not np.isfinite(box).all():
                continue
            if not np.isfinite(confidence) or box[2] <= box[0] or box[3] <= box[1]:
                continue

            class_id = self._class_ids.setdefault(class_name, len(self._class_ids))
            self._class_names[class_id] = class_name
            boxes.append(box)
            confidences.append(confidence)
            classes.append(class_id)

        return _DetectionBatch(
            np.asarray(boxes, dtype=np.float32).reshape((-1, 4)),
            np.asarray(confidences, dtype=np.float32),
            np.asarray(classes, dtype=np.float32),
        )

    def update(self, detections: Sequence[dict] | None) -> list[dict]:
        """Update tracks with the current frame's detector output.

        Args:
            detections: List of dictionaries returned by ``YOLODetector.detect``.
                Pass an empty list when the current frame contains no detections.

        Returns:
            Current visible tracks, each with ``track_id``, ``class_name``,
            ``confidence``, and ``bounding_box``.
        """
        batch = self._batch_from_detections(detections or [])
        tracked = self._tracker.update(batch)

        tracks: list[dict] = []
        for row in tracked:
            class_id = int(row[6])
            tracks.append(
                {
                    "track_id": int(row[4]),
                    "class_name": self._class_names.get(class_id, "object"),
                    "confidence": float(row[5]),
                    "bounding_box": [float(value) for value in row[:4]],
                }
            )
        return tracks

    def reset(self) -> None:
        """Clear active tracker state and class-name mappings."""
        self._tracker.reset()
        self._class_ids.clear()
        self._class_names.clear()
