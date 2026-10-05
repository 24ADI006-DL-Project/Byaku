from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Any, Literal, Protocol


DistanceCategory = Literal["very_close", "close", "moderate", "far"]


@dataclass(frozen=True)
class RelativeDepthThresholds:
    """Upper bounds for relative-depth categories; larger values mean farther."""

    very_close_max: float = 0.25
    close_max: float = 0.5
    moderate_max: float = 0.75

    def __post_init__(self) -> None:
        values = (self.very_close_max, self.close_max, self.moderate_max)
        if not all(isfinite(value) and value >= 0 for value in values):
            raise ValueError("depth thresholds must be finite and non-negative")
        if not self.very_close_max < self.close_max < self.moderate_max:
            raise ValueError("depth thresholds must be strictly increasing")


class DistanceEstimator(Protocol):
    """Strategy interface for mapping a depth measurement to a category."""

    def estimate_category(self, relative_depth: Any) -> DistanceCategory | None:
        """Return a qualitative category, or None when depth is unavailable."""


class RelativeDepthEstimator:
    """Classify relative depth without interpreting it as metric distance.

    Defaults assume normalized relative-depth values where larger values are
    farther away. Configure thresholds to match the depth model's output scale.
    """

    def __init__(self, thresholds: RelativeDepthThresholds | None = None) -> None:
        self.thresholds = thresholds or RelativeDepthThresholds()

    def estimate_category(self, relative_depth: Any) -> DistanceCategory | None:
        try:
            value = float(relative_depth)
        except (TypeError, ValueError):
            return None

        if not isfinite(value) or value < 0:
            return None
        if value <= self.thresholds.very_close_max:
            return "very_close"
        if value <= self.thresholds.close_max:
            return "close"
        if value <= self.thresholds.moderate_max:
            return "moderate"
        return "far"