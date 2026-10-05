from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from math import isfinite
from typing import Any

from reasoning.priority import PriorityScorer


@dataclass(frozen=True)
class EventThresholds:
    cooldown_frames: int = 10
    disappearance_frames: int = 2
    significant_depth_delta: float = 0.2

    def __post_init__(self) -> None:
        if self.cooldown_frames < 0:
            raise ValueError("cooldown_frames must be non-negative")
        if self.disappearance_frames < 1:
            raise ValueError("disappearance_frames must be at least 1")
        if not isfinite(self.significant_depth_delta) or self.significant_depth_delta < 0:
            raise ValueError("significant_depth_delta must be finite and non-negative")


class EventDetector:
    """Emit deterministic scene-change decisions without generating speech."""

    _distance_rank = {"very_close": 0, "close": 1, "moderate": 2, "far": 3}
    _positions = {"left", "center", "right"}

    def __init__(
        self,
        thresholds: EventThresholds | None = None,
        priority_scorer: PriorityScorer | None = None,
    ) -> None:
        self.thresholds = thresholds or EventThresholds()
        self.priority_scorer = priority_scorer or PriorityScorer()
        self._frame_number = 0
        self._states: dict[Any, dict[str, Any]] = {}
        self._last_event_frame: dict[tuple[Any, str], int] = {}
        self._last_announced_frame: dict[Any, int] = {}

    def update(self, spatial_objects: Sequence[Mapping[str, Any]] | None) -> list[dict[str, Any]]:
        """Compare one frame with the previous state and return event decisions."""
        self._frame_number += 1
        self._prune_expired()
        current = self._objects_by_track(spatial_objects)
        events: list[dict[str, Any]] = []

        for track_id, spatial_object in current.items():
            state = self._states.get(track_id)
            if state is None or not state["present"]:
                event = self._make_event(
                    "new_object",
                    spatial_object,
                    newly_detected=True,
                    significant=True,
                )
                if event is not None:
                    events.append(event)
                self._states[track_id] = {
                    "object": spatial_object,
                    "present": True,
                    "missing_frames": 0,
                    "disappeared_frame": None,
                }
                continue

            previous = state["object"]
            events.extend(self._detect_changes(previous, spatial_object))
            state.update(object=spatial_object, missing_frames=0)

        for track_id, state in list(self._states.items()):
            if track_id in current or not state["present"]:
                continue
            state["missing_frames"] += 1
            if state["missing_frames"] >= self.thresholds.disappearance_frames:
                event = self._make_event("object_disappeared", state["object"])
                if event is not None:
                    events.append(event)
                state.update(present=False, disappeared_frame=self._frame_number)

        return events

    def _detect_changes(
        self,
        previous: Mapping[str, Any],
        current: Mapping[str, Any],
    ) -> list[dict[str, Any]]:
        track_id = current["track_id"]
        events: list[dict[str, Any]] = []
        old_category = previous.get("distance_category")
        new_category = current.get("distance_category")
        distance_changed = old_category != new_category

        old_rank = self._distance_rank.get(old_category)
        new_rank = self._distance_rank.get(new_category)
        if distance_changed and old_rank is not None and new_rank is not None:
            event_type = "object_approaching" if new_rank < old_rank else "distance_category_changed"
            event = self._make_event(
                event_type,
                current,
                significant_movement=True,
                details={
                    "previous_distance_category": old_category,
                    "distance_category": new_category,
                    "significant": True,
                },
            )
            if event is not None:
                events.append(event)

        old_position = previous.get("horizontal_position")
        new_position = current.get("horizontal_position")
        if (
            old_position != new_position
            and old_position in self._positions
            and new_position in self._positions
        ):
            event = self._make_event(
                "position_changed",
                current,
                significant_position_change=True,
                details={
                    "previous_position": old_position,
                    "horizontal_position": new_position,
                    "significant": True,
                },
            )
            if event is not None:
                events.append(event)

        if not distance_changed and self._depth_changed_significantly(previous, current):
            event = self._make_event(
                "significant_spatial_change",
                current,
                significant_movement=True,
                details={"significant": True},
            )
            if event is not None:
                events.append(event)

        return events

    def _make_event(
        self,
        event_type: str,
        spatial_object: Mapping[str, Any],
        *,
        newly_detected: bool = False,
        significant_movement: bool = False,
        significant_position_change: bool = False,
        significant: bool = False,
        details: Mapping[str, Any] | None = None,
    ) -> dict[str, Any] | None:
        track_id = spatial_object["track_id"]
        key = (track_id, event_type)
        last_frame = self._last_event_frame.get(key)
        if last_frame is not None and self._frame_number - last_frame <= self.thresholds.cooldown_frames:
            return None

        last_announced = self._last_announced_frame.get(track_id)
        recently_announced = (
            last_announced is not None
            and self._frame_number - last_announced <= self.thresholds.cooldown_frames
        )
        assessment = self.priority_scorer.assess(
            spatial_object,
            newly_detected=newly_detected,
            significant_movement=significant_movement,
            significant_position_change=significant_position_change,
            recently_announced=recently_announced,
        )
        event = {
            "event_type": event_type,
            "track_id": track_id,
            "class_name": spatial_object.get("class_name", "object"),
            "priority": assessment["priority"],
            "score": assessment["score"],
            "message_required": assessment["should_announce"],
            "significant": significant or significant_movement or significant_position_change,
        }
        if details:
            event.update(details)

        self._last_event_frame[key] = self._frame_number
        if event["message_required"]:
            self._last_announced_frame[track_id] = self._frame_number
        return event

    def _prune_expired(self) -> None:
        cooldown = self.thresholds.cooldown_frames
        self._last_event_frame = {
            key: frame for key, frame in self._last_event_frame.items()
            if self._frame_number - frame <= cooldown
        }
        self._last_announced_frame = {
            track_id: frame for track_id, frame in self._last_announced_frame.items()
            if self._frame_number - frame <= cooldown
        }
        for track_id, state in list(self._states.items()):
            disappeared_frame = state["disappeared_frame"]
            if disappeared_frame is not None and self._frame_number - disappeared_frame > cooldown:
                del self._states[track_id]

    @staticmethod
    def _objects_by_track(
        spatial_objects: Sequence[Mapping[str, Any]] | None,
    ) -> dict[Any, dict[str, Any]]:
        objects: dict[Any, dict[str, Any]] = {}
        for spatial_object in spatial_objects or []:
            if not isinstance(spatial_object, Mapping) or "track_id" not in spatial_object:
                continue
            track_id = spatial_object["track_id"]
            try:
                hash(track_id)
            except TypeError:
                continue
            objects[track_id] = dict(spatial_object)
        return objects

    def _depth_changed_significantly(
        self,
        previous: Mapping[str, Any],
        current: Mapping[str, Any],
    ) -> bool:
        try:
            old_depth = float(previous.get("relative_depth"))
            new_depth = float(current.get("relative_depth"))
        except (TypeError, ValueError):
            return False
        return (
            isfinite(old_depth)
            and isfinite(new_depth)
            and abs(new_depth - old_depth) >= self.thresholds.significant_depth_delta
        )