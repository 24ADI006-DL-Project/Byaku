from collections.abc import Iterable, Mapping, Sequence
from copy import deepcopy
from numbers import Integral
import math
import time
from typing import Any

from .spatial_relation_engine import DEPTH_CATEGORIES


class SceneMemory:
    """Keep bounded recent observations for tracked objects across frames."""

    def __init__(
        self,
        stale_after_frames: int = 5,
        history_size: int = 20,
        stale_retention_frames: int = 300,
    ) -> None:
        if not isinstance(stale_after_frames, Integral) or isinstance(stale_after_frames, bool) or stale_after_frames < 1:
            raise ValueError("stale_after_frames must be a positive integer")
        if not isinstance(history_size, Integral) or isinstance(history_size, bool) or history_size < 1:
            raise ValueError("history_size must be a positive integer")
        if (
            not isinstance(stale_retention_frames, Integral)
            or isinstance(stale_retention_frames, bool)
            or stale_retention_frames < stale_after_frames
        ):
            raise ValueError("stale_retention_frames must be an integer at least stale_after_frames")

        self.stale_after_frames = int(stale_after_frames)
        self.history_size = int(history_size)
        self.stale_retention_frames = int(stale_retention_frames)
        self._frame_number = 0
        self._objects: dict[str, dict[str, Any]] = {}
        self._last_warnings: list[str] = []

    @staticmethod
    def _valid_id(value: Any) -> bool:
        if isinstance(value, bool):
            return False
        if isinstance(value, Integral):
            return True
        if isinstance(value, str):
            return bool(value.strip())
        if isinstance(value, float):
            return math.isfinite(value)
        return False

    @staticmethod
    def _canonical_track_id(value: Any) -> str:
        if isinstance(value, str) and value.startswith("track:"):
            return value
        return f"track:{value}"

    @staticmethod
    def _coerce_box(value: Any) -> tuple[float, float, float, float] | None:
        try:
            coordinates = tuple(float(item) for item in value)
        except (TypeError, ValueError, OverflowError):
            return None
        if len(coordinates) != 4 or not all(math.isfinite(item) for item in coordinates):
            return None
        x1, y1, x2, y2 = coordinates
        if x1 >= x2 or y1 >= y2:
            return None
        return x1, y1, x2, y2

    @staticmethod
    def _coerce_center(value: Any, box: tuple[float, float, float, float]) -> tuple[float, float]:
        try:
            center = tuple(float(item) for item in value)
        except (TypeError, ValueError, OverflowError):
            center = ()
        if len(center) == 2 and all(math.isfinite(item) for item in center):
            return center[0], center[1]
        return (box[0] + box[2]) / 2, (box[1] + box[3]) / 2

    def _normalize_frame_number(self, frame_number: Any) -> int:
        if isinstance(frame_number, Integral) and not isinstance(frame_number, bool):
            candidate = int(frame_number)
            if candidate > self._frame_number:
                self._frame_number = candidate
                return candidate
            self._last_warnings.append(
                f"frame number {candidate} is not newer than {self._frame_number}; advanced automatically"
            )
        elif frame_number is not None:
            self._last_warnings.append("invalid frame number; advanced automatically")

        self._frame_number += 1
        return self._frame_number

    def _normalize_observations(self, frame_data: Any) -> list[Mapping[str, Any]]:
        if frame_data is None:
            return []
        if isinstance(frame_data, Mapping):
            if "objects" in frame_data:
                frame_data = frame_data["objects"]
            else:
                frame_data = [frame_data]
        if isinstance(frame_data, (str, bytes)) or not isinstance(frame_data, Iterable):
            self._last_warnings.append("frame observations must be an iterable of object mappings")
            return []

        observations = []
        for index, item in enumerate(frame_data):
            if not isinstance(item, Mapping):
                self._last_warnings.append(f"observation {index} is not a mapping and was skipped")
                continue
            observations.append(item)
        return observations

    def _identity_for(
        self,
        observation: Mapping[str, Any],
        frame_number: int,
        observation_index: int,
    ) -> tuple[str, Any, bool]:
        track_id = observation.get("track_id")
        object_id = observation.get("object_id")

        if track_id is None and isinstance(object_id, str):
            if object_id.startswith("track:"):
                track_id = object_id[len("track:"):]
            elif object_id.startswith("temporary:"):
                return object_id, None, True
            elif object_id.startswith("frame:"):
                suffix = object_id[len("frame:"):]
                return f"temporary:{frame_number}:{suffix or observation_index}", None, True
            elif not object_id.startswith(("frame:", "temporary:")):
                track_id = object_id

        if track_id is None:
            return f"temporary:{frame_number}:{observation_index}", None, True
        if not self._valid_id(track_id):
            self._last_warnings.append(
                f"observation {observation_index} has an invalid tracking ID; assigned a temporary ID"
            )
            return f"temporary:{frame_number}:{observation_index}", None, True
        return self._canonical_track_id(track_id), track_id, False

    def update(
        self,
        frame_data: Any,
        *,
        frame_number: int | None = None,
        timestamp: float | str | None = None,
    ) -> dict[str, Any]:
        """Update memory from object records or a spatial output's ``objects`` list.

        Tracked IDs persist across frames. Records with only ``frame:<index>`` IDs
        are treated as per-frame observations and are not carried across frames.
        """
        self._last_warnings = []
        current_frame = self._normalize_frame_number(frame_number)
        observed_ids: set[str] = set()
        observations = self._normalize_observations(frame_data)
        frame_timestamp = time.time() if timestamp is None else timestamp

        for index, observation in enumerate(observations):
            raw_box = observation.get("bounding_box", observation.get("bbox"))
            box = self._coerce_box(raw_box)
            if box is None:
                self._last_warnings.append(f"observation {index} has an invalid bounding box and was skipped")
                continue

            class_name = observation.get("class_name", observation.get("class", observation.get("name")))
            if not isinstance(class_name, str) or not class_name.strip():
                self._last_warnings.append(f"observation {index} has no valid class name and was skipped")
                continue

            object_id, track_id, is_temporary = self._identity_for(observation, current_frame, index)
            if is_temporary and object_id in self._objects:
                existing = self._objects[object_id]
                if existing["last_seen_frame"] != current_frame:
                    self._last_warnings.append(
                        f"temporary ID {object_id!r} was reused across frames; assigned a frame-local ID"
                    )
                    object_id = f"temporary:{current_frame}:{index}"
                    suffix = 2
                    while object_id in self._objects or object_id in observed_ids:
                        object_id = f"temporary:{current_frame}:{index}:{suffix}"
                        suffix += 1
            if object_id in observed_ids:
                if is_temporary:
                    message = f"duplicate temporary ID {object_id!r} in frame {current_frame}; retained the first observation"
                else:
                    message = f"duplicate tracking ID {track_id!r} in frame {current_frame}; retained the first observation"
                self._last_warnings.append(message)
                continue
            observed_ids.add(object_id)

            raw_center = observation.get("center")
            center = self._coerce_center(raw_center, box)
            if raw_center is not None:
                try:
                    supplied_center = tuple(float(value) for value in raw_center)
                    center_is_valid = len(supplied_center) == 2 and all(math.isfinite(value) for value in supplied_center)
                except (TypeError, ValueError, OverflowError):
                    center_is_valid = False
                if not center_is_valid:
                    self._last_warnings.append(f"observation {index} has an invalid center; derived it from the box")

            depth_value = observation.get("relative_depth_value")
            if depth_value is not None:
                try:
                    depth_value = float(depth_value)
                except (TypeError, ValueError, OverflowError):
                    depth_value = None
                    self._last_warnings.append(f"observation {index} has invalid relative depth; stored as missing")
                else:
                    if not math.isfinite(depth_value):
                        depth_value = None
                        self._last_warnings.append(f"observation {index} has non-finite relative depth; stored as missing")

            observation_record = {
                "frame_number": current_frame,
                "timestamp": frame_timestamp,
                "class_name": class_name,
                "bounding_box": box,
                "center": center,
                "relative_depth_value": depth_value,
            }
            depth_category = observation.get("depth_category")
            if isinstance(depth_category, str) and depth_category in DEPTH_CATEGORIES:
                observation_record["depth_category"] = depth_category

            confidence = observation.get("confidence")
            if confidence is not None:
                try:
                    confidence = float(confidence)
                except (TypeError, ValueError, OverflowError):
                    confidence = None
                    self._last_warnings.append(f"observation {index} has invalid confidence; stored as missing")
                else:
                    if not math.isfinite(confidence):
                        confidence = None
                        self._last_warnings.append(
                            f"observation {index} has non-finite confidence; stored as missing"
                        )
                if confidence is not None:
                    observation_record["confidence"] = confidence

            horizontal_pos = observation.get("horizontal_position")
            if horizontal_pos in {"left", "center", "right"}:
                observation_record["horizontal_position"] = horizontal_pos
            vertical_pos = observation.get("vertical_position")
            if vertical_pos in {"above", "middle", "below"}:
                observation_record["vertical_position"] = vertical_pos

            if object_id not in self._objects:
                new_entry = {
                    "object_id": object_id,
                    "track_id": track_id,
                    "is_temporary": is_temporary,
                    "class_name": class_name,
                    "bounding_box": box,
                    "center": center,
                    "relative_depth_value": depth_value,
                    "last_seen_frame": current_frame,
                    "last_seen_timestamp": frame_timestamp,
                    "status": "visible",
                    "observations": [],
                }
                if "horizontal_position" in observation_record:
                    new_entry["horizontal_position"] = observation_record["horizontal_position"]
                if "vertical_position" in observation_record:
                    new_entry["vertical_position"] = observation_record["vertical_position"]
                if "depth_category" in observation_record:
                    new_entry["depth_category"] = observation_record["depth_category"]
                self._objects[object_id] = new_entry

            entry = self._objects[object_id]
            entry.update(
                {
                    "class_name": class_name,
                    "bounding_box": box,
                    "center": center,
                    "relative_depth_value": depth_value,
                    "last_seen_frame": current_frame,
                    "last_seen_timestamp": frame_timestamp,
                    "status": "visible",
                }
            )
            if "horizontal_position" in observation_record:
                entry["horizontal_position"] = observation_record["horizontal_position"]
            if "vertical_position" in observation_record:
                entry["vertical_position"] = observation_record["vertical_position"]
            if "depth_category" in observation_record:
                entry["depth_category"] = observation_record["depth_category"]
            else:
                entry.pop("depth_category", None)
            if confidence is None:
                entry.pop("confidence", None)
            else:
                entry["confidence"] = confidence
            entry["observations"].append(observation_record)
            if len(entry["observations"]) > self.history_size:
                del entry["observations"][:-self.history_size]

        for object_id, entry in list(self._objects.items()):
            if entry["is_temporary"]:
                if object_id not in observed_ids and entry["last_seen_frame"] != current_frame:
                    del self._objects[object_id]
                continue
            if object_id in observed_ids:
                entry["status"] = "visible"
                continue

            frames_missing = current_frame - entry["last_seen_frame"]
            if frames_missing >= self.stale_after_frames:
                if frames_missing > self.stale_retention_frames:
                    del self._objects[object_id]
                    continue
                entry["status"] = "stale"
            else:
                entry["status"] = "missing"

        return {
            "frame_number": current_frame,
            "visible_objects": self.get_visible_objects(),
            "temporarily_missing_objects": self.get_temporarily_missing_objects(),
            "stale_objects": self.get_stale_objects(),
            "warnings": list(self._last_warnings),
        }

    def _objects_with_status(self, status: str) -> list[dict[str, Any]]:
        return [
            deepcopy(entry)
            for entry in self._objects.values()
            if entry["status"] == status and not entry["is_temporary"]
        ]

    def get_visible_objects(self) -> list[dict[str, Any]]:
        """Return only objects observed in the latest update, including per-frame IDs."""
        return [deepcopy(entry) for entry in self._objects.values() if entry["status"] == "visible"]

    def get_temporarily_missing_objects(self) -> list[dict[str, Any]]:
        """Return tracked objects missing for fewer than ``stale_after_frames`` frames."""
        return self._objects_with_status("missing")

    def get_stale_objects(self) -> list[dict[str, Any]]:
        """Return tracked objects whose last observation exceeds the age threshold."""
        return self._objects_with_status("stale")

    @staticmethod
    def _lookup_id(value: Any) -> str | None:
        if not SceneMemory._valid_id(value):
            return None
        if isinstance(value, str) and value.startswith(("track:", "frame:", "temporary:")):
            return value
        return f"track:{value}"

    def get_object(self, tracking_id: Any) -> dict[str, Any] | None:
        """Return a tracked object by ID, or a per-frame object by its full object ID."""
        object_id = self._lookup_id(tracking_id)
        if object_id is None:
            return None
        entry = self._objects.get(object_id)
        return deepcopy(entry) if entry is not None else None

    def get_recent_observations(self, tracking_id: Any, limit: int | None = None) -> list[dict[str, Any]]:
        """Return most recent observations for a tracked or temporary object ID."""
        entry = self.get_object(tracking_id)
        if entry is None:
            return []
        observations: Sequence[dict[str, Any]] = entry["observations"]
        if limit is None:
            return deepcopy(list(observations))
        if not isinstance(limit, Integral) or isinstance(limit, bool) or limit <= 0:
            return []
        return deepcopy(list(observations[-int(limit):]))