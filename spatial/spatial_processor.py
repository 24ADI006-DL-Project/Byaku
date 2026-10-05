from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, TypedDict, cast

from spatial.distance import DistanceCategory, DistanceEstimator, RelativeDepthEstimator
from spatial.position import HorizontalPosition, HorizontalPositionEstimator
from spatial.spatial_object import SpatialObject


class ProcessedSpatialObject(SpatialObject):
    horizontal_position: HorizontalPosition | None
    distance_category: DistanceCategory | None


class SpatialProcessor:
    """Add horizontal position and qualitative distance to Module 5 objects."""

    def __init__(
        self,
        position_estimator: HorizontalPositionEstimator | None = None,
        distance_estimator: DistanceEstimator | None = None,
    ) -> None:
        self.position_estimator = position_estimator or HorizontalPositionEstimator()
        self.distance_estimator = distance_estimator or RelativeDepthEstimator()

    def process(
        self,
        enriched_objects: Sequence[Mapping[str, Any]] | None,
        image_shape: Sequence[int] | None,
    ) -> list[ProcessedSpatialObject]:
        """Return copied objects with position and relative-depth category fields."""
        if enriched_objects is None:
            return []

        processed: list[ProcessedSpatialObject] = []
        for enriched in enriched_objects:
            if not isinstance(enriched, Mapping):
                continue

            item = dict(enriched)
            item["horizontal_position"] = self.position_estimator.classify(
                item.get("bounding_box"), image_shape
            )
            item["distance_category"] = self.distance_estimator.estimate_category(
                item.get("relative_depth")
            )
            processed.append(cast(ProcessedSpatialObject, item))

        return processed