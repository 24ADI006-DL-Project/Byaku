from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from math import isfinite
from typing import Any, Literal


Priority = Literal["low", "medium", "high"]


@dataclass(frozen=True)
class PriorityConfig:
    distance_points: Mapping[str, float] = field(
        default_factory=lambda: {"very_close": 4, "close": 2, "moderate": 1, "far": 0}
    )
    important_classes: frozenset[str] = frozenset(
        {"person", "vehicle", "car", "stairs", "door", "wall", "obstacle"}
    )
    important_class_bonus: float = 3
    newly_detected_bonus: float = 1
    significant_movement_bonus: float = 2
    position_change_bonus: float = 2
    recently_announced_penalty: float = 3
    high_threshold: float = 6
    medium_threshold: float = 3
    announcement_threshold: float = 3

    def __post_init__(self) -> None:
        values = (
            *self.distance_points.values(),
            self.important_class_bonus,
            self.newly_detected_bonus,
            self.significant_movement_bonus,
            self.position_change_bonus,
            self.recently_announced_penalty,
            self.high_threshold,
            self.medium_threshold,
            self.announcement_threshold,
        )
        if not all(isfinite(value) and value >= 0 for value in values):
            raise ValueError("priority points and thresholds must be finite and non-negative")
        if self.high_threshold < self.medium_threshold:
            raise ValueError("high_threshold must be at least medium_threshold")


class PriorityScorer:
    """Calculate an explainable priority score and announce decision."""

    def __init__(self, config: PriorityConfig | None = None) -> None:
        self.config = config or PriorityConfig()

    def score(
        self,
        spatial_object: Mapping[str, Any],
        *,
        newly_detected: bool = False,
        significant_movement: bool = False,
        significant_position_change: bool = False,
        recently_announced: bool = False,
    ) -> float:
        config = self.config
        category = spatial_object.get("distance_category")
        score = float(config.distance_points.get(category, 0))

        class_name = str(spatial_object.get("class_name", "")).strip().lower()
        important_classes = {name.lower() for name in config.important_classes}
        if class_name in important_classes:
            score += config.important_class_bonus
        if newly_detected:
            score += config.newly_detected_bonus
        if significant_movement:
            score += config.significant_movement_bonus
        if significant_position_change:
            score += config.position_change_bonus
        if recently_announced:
            score -= config.recently_announced_penalty

        return max(0.0, score)

    def assess(
        self,
        spatial_object: Mapping[str, Any],
        *,
        newly_detected: bool = False,
        significant_movement: bool = False,
        significant_position_change: bool = False,
        recently_announced: bool = False,
    ) -> dict[str, Any]:
        score = self.score(
            spatial_object,
            newly_detected=newly_detected,
            significant_movement=significant_movement,
            significant_position_change=significant_position_change,
            recently_announced=recently_announced,
        )
        if score >= self.config.high_threshold:
            priority: Priority = "high"
        elif score >= self.config.medium_threshold:
            priority = "medium"
        else:
            priority = "low"

        return {
            "score": score,
            "priority": priority,
            "should_announce": score >= self.config.announcement_threshold,
        }