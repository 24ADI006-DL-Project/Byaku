from __future__ import annotations

from collections.abc import Mapping, Sequence
from math import ceil, floor, isfinite
from typing import Any

import numpy as np

from spatial.spatial_object import SpatialObject


class DepthIntegrator:
    """Attach a robust relative-depth estimate to tracked objects."""

    def integrate(
        self,
        tracked_objects: Sequence[Mapping[str, Any]] | None,
        depth_map: np.ndarray | None,
        image_shape: Sequence[int] | None = None,
    ) -> list[SpatialObject]:
        """Return copies of tracked objects with median depth from their boxes.

        ``image_shape`` can be the source frame shape (height, width[, channels]);
        when supplied, a mismatched depth map is treated as unavailable.
        """
        objects = [] if tracked_objects is None else tracked_objects
        valid_map = self._valid_depth_map(depth_map, image_shape)
        enriched: list[SpatialObject] = []

        for tracked in objects:
            if not isinstance(tracked, Mapping):
                continue

            item = dict(tracked)
            item["relative_depth"] = self._median_for_box(item.get("bounding_box"), valid_map)
            enriched.append(item)  # type: ignore[arg-type]

        return enriched

    @staticmethod
    def _valid_depth_map(
        depth_map: np.ndarray | None,
        image_shape: Sequence[int] | None,
    ) -> np.ndarray | None:
        if depth_map is None:
            return None

        try:
            values = np.asarray(depth_map)
        except (TypeError, ValueError):
            return None

        if values.ndim != 2 or not np.issubdtype(values.dtype, np.number):
            return None

        if image_shape is not None:
            try:
                expected_shape = tuple(int(dimension) for dimension in image_shape[:2])
            except (TypeError, ValueError):
                return None
            if len(expected_shape) != 2 or values.shape != expected_shape:
                return None

        return values

    @staticmethod
    def _median_for_box(bounding_box: Any, depth_map: np.ndarray | None) -> float | None:
        if depth_map is None or not isinstance(bounding_box, (list, tuple, np.ndarray)):
            return None

        try:
            if len(bounding_box) != 4:
                return None
            x1, y1, x2, y2 = (float(value) for value in bounding_box)
        except (TypeError, ValueError):
            return None

        if not all(isfinite(value) for value in (x1, y1, x2, y2)) or x2 <= x1 or y2 <= y1:
            return None

        height, width = depth_map.shape
        left = max(0, min(width, floor(x1)))
        top = max(0, min(height, floor(y1)))
        right = max(0, min(width, ceil(x2)))
        bottom = max(0, min(height, ceil(y2)))
        if left >= right or top >= bottom:
            return None

        region = depth_map[top:bottom, left:right]
        finite_values = region[np.isfinite(region)]
        if finite_values.size == 0:
            return None

        return float(np.median(finite_values))