from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Any, Literal, Sequence


HorizontalPosition = Literal["left", "center", "right"]


@dataclass(frozen=True)
class PositionThresholds:
    """Normalized horizontal center boundaries for left and right regions."""

    left_boundary: float = 1 / 3
    right_boundary: float = 2 / 3

    def __post_init__(self) -> None:
        if not (
            isfinite(self.left_boundary)
            and isfinite(self.right_boundary)
            and 0 <= self.left_boundary < self.right_boundary <= 1
        ):
            raise ValueError("position boundaries must satisfy 0 <= left < right <= 1")


class HorizontalPositionEstimator:
    """Classify a bounding-box center within a frame as left, center, or right."""

    def __init__(self, thresholds: PositionThresholds | None = None) -> None:
        self.thresholds = thresholds or PositionThresholds()

    def classify(
        self,
        bounding_box: Any,
        image_shape: Sequence[int] | None,
    ) -> HorizontalPosition | None:
        dimensions = self._image_dimensions(image_shape)
        if dimensions is None or not isinstance(bounding_box, (list, tuple)):
            return None

        try:
            if len(bounding_box) != 4:
                return None
            x1, y1, x2, y2 = (float(value) for value in bounding_box)
        except (TypeError, ValueError):
            return None

        if not all(isfinite(value) for value in (x1, y1, x2, y2)) or x2 <= x1 or y2 <= y1:
            return None

        height, width = dimensions
        clipped_left = max(0.0, min(float(width), x1))
        clipped_right = max(0.0, min(float(width), x2))
        clipped_top = max(0.0, min(float(height), y1))
        clipped_bottom = max(0.0, min(float(height), y2))
        if clipped_left >= clipped_right or clipped_top >= clipped_bottom:
            return None

        center_fraction = ((clipped_left + clipped_right) / 2) / width
        if center_fraction < self.thresholds.left_boundary:
            return "left"
        if center_fraction > self.thresholds.right_boundary:
            return "right"
        return "center"

    @staticmethod
    def _image_dimensions(image_shape: Sequence[int] | None) -> tuple[int, int] | None:
        if image_shape is None:
            return None
        try:
            if len(image_shape) < 2:
                return None
            height, width = int(image_shape[0]), int(image_shape[1])
        except (TypeError, ValueError, IndexError):
            return None
        if height <= 0 or width <= 0:
            return None
        return height, width