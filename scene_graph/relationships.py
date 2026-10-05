from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from math import isfinite
from typing import Any, Literal, TypedDict


RelationshipType = Literal[
    "left_of",
    "right_of",
    "near",
    "far",
    "in_front_of",
    "behind",
    "next_to",
]


class Relationship(TypedDict):
    source: int
    relationship: RelationshipType
    target: int


@dataclass(frozen=True)
class RelationshipThresholds:
    """Limits for rule-based relationships; depth ranks follow Module 6."""

    near_max_category_rank: int = 1
    near_max_category_gap: int = 1
    far_min_category_gap: int = 2
    front_depth_delta: float = 0.1
    next_to_pixel_gap: float = 40.0

    def __post_init__(self) -> None:
        if self.near_max_category_rank < 0 or self.near_max_category_gap < 0:
            raise ValueError("near relationship thresholds must be non-negative")
        if self.far_min_category_gap < 1:
            raise ValueError("far_min_category_gap must be at least 1")
        if self.near_max_category_gap >= self.far_min_category_gap:
            raise ValueError("near category gap must be smaller than far category gap")
        if not isfinite(self.front_depth_delta) or self.front_depth_delta < 0:
            raise ValueError("front_depth_delta must be finite and non-negative")
        if not isfinite(self.next_to_pixel_gap) or self.next_to_pixel_gap < 0:
            raise ValueError("next_to_pixel_gap must be finite and non-negative")


class RelationshipGenerator:
    """Generate explainable relationships from Module 6 spatial fields."""

    _position_rank = {"left": 0, "center": 1, "right": 2}
    _distance_rank = {"very_close": 0, "close": 1, "moderate": 2, "far": 3}

    def __init__(self, thresholds: RelationshipThresholds | None = None) -> None:
        self.thresholds = thresholds or RelationshipThresholds()

    def generate(self, objects: Sequence[Mapping[str, Any]]) -> list[Relationship]:
        relationships: list[Relationship] = []
        valid_objects = [item for item in objects if self._has_track_id(item)]

        for index, first in enumerate(valid_objects):
            for second in valid_objects[index + 1 :]:
                first_id = first["track_id"]
                second_id = second["track_id"]
                if first_id == second_id:
                    continue

                self._add_horizontal(relationships, first, second)
                self._add_depth_category(relationships, first, second)
                self._add_depth_order(relationships, first, second)
                self._add_next_to(relationships, first, second)

        return relationships

    @staticmethod
    def _has_track_id(item: Mapping[str, Any]) -> bool:
        try:
            hash(item.get("track_id"))
            return item.get("track_id") is not None
        except TypeError:
            return False

    def _add_horizontal(
        self,
        relationships: list[Relationship],
        first: Mapping[str, Any],
        second: Mapping[str, Any],
    ) -> None:
        first_rank = self._position_rank.get(first.get("horizontal_position"))
        second_rank = self._position_rank.get(second.get("horizontal_position"))
        if first_rank is None or second_rank is None or first_rank == second_rank:
            return

        left, right = (first, second) if first_rank < second_rank else (second, first)
        self._append(relationships, left["track_id"], "left_of", right["track_id"])
        self._append(relationships, right["track_id"], "right_of", left["track_id"])

    def _add_depth_category(
        self,
        relationships: list[Relationship],
        first: Mapping[str, Any],
        second: Mapping[str, Any],
    ) -> None:
        first_rank = self._distance_rank.get(first.get("distance_category"))
        second_rank = self._distance_rank.get(second.get("distance_category"))
        if first_rank is None or second_rank is None:
            return

        gap = abs(first_rank - second_rank)
        category = None
        if (
            max(first_rank, second_rank) <= self.thresholds.near_max_category_rank
            and gap <= self.thresholds.near_max_category_gap
        ):
            category = "near"
        elif gap >= self.thresholds.far_min_category_gap:
            category = "far"

        if category is not None:
            source, target = self._stable_pair(first["track_id"], second["track_id"])
            self._append(relationships, source, category, target)

    def _add_depth_order(
        self,
        relationships: list[Relationship],
        first: Mapping[str, Any],
        second: Mapping[str, Any],
    ) -> None:
        try:
            first_depth = float(first.get("relative_depth"))
            second_depth = float(second.get("relative_depth"))
        except (TypeError, ValueError):
            return
        if not isfinite(first_depth) or not isfinite(second_depth):
            return

        if first_depth + self.thresholds.front_depth_delta < second_depth:
            front, behind = first, second
        elif second_depth + self.thresholds.front_depth_delta < first_depth:
            front, behind = second, first
        else:
            return

        self._append(relationships, front["track_id"], "in_front_of", behind["track_id"])
        self._append(relationships, behind["track_id"], "behind", front["track_id"])

    def _add_next_to(
        self,
        relationships: list[Relationship],
        first: Mapping[str, Any],
        second: Mapping[str, Any],
    ) -> None:
        first_box = self._box(first.get("bounding_box"))
        second_box = self._box(second.get("bounding_box"))
        if first_box is None or second_box is None:
            return

        if first_box[2] <= second_box[0]:
            horizontal_gap = second_box[0] - first_box[2]
        elif second_box[2] <= first_box[0]:
            horizontal_gap = first_box[0] - second_box[2]
        else:
            return

        vertical_overlap = min(first_box[3], second_box[3]) - max(first_box[1], second_box[1])
        if horizontal_gap > self.thresholds.next_to_pixel_gap or vertical_overlap <= 0:
            return

        source, target = self._stable_pair(first["track_id"], second["track_id"])
        self._append(relationships, source, "next_to", target)

    @staticmethod
    def _box(value: Any) -> tuple[float, float, float, float] | None:
        if not isinstance(value, (list, tuple)) or len(value) != 4:
            return None
        try:
            box = tuple(float(item) for item in value)
        except (TypeError, ValueError):
            return None
        if not all(isfinite(item) for item in box) or box[2] <= box[0] or box[3] <= box[1]:
            return None
        return box  # type: ignore[return-value]

    @staticmethod
    def _stable_pair(first_id: int, second_id: int) -> tuple[int, int]:
        first_key = (type(first_id).__name__, repr(first_id))
        second_key = (type(second_id).__name__, repr(second_id))
        return (first_id, second_id) if first_key <= second_key else (second_id, first_id)

    @staticmethod
    def _append(
        relationships: list[Relationship],
        source: int,
        relationship: RelationshipType,
        target: int,
    ) -> None:
        if source != target:
            relationships.append({"source": source, "relationship": relationship, "target": target})