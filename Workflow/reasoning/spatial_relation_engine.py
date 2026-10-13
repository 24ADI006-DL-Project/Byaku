from collections.abc import Iterable, Mapping, Sequence
from itertools import count
from numbers import Integral
from typing import Any

import numpy as np


_FRAME_ID_COUNTER = count(1)
DEFAULT_DEPTH_DIFFERENCE_THRESHOLD = 0.05


def image_shape_from_frame_size(frame_size: Mapping[str, Any]) -> tuple[int, int]:
    """Convert external ``{width, height}`` data to canonical ``(height, width)``."""
    if not isinstance(frame_size, Mapping):
        raise ValueError("frame_size must be a mapping with width and height")
    width = frame_size.get("width")
    height = frame_size.get("height")
    if (
        isinstance(width, bool)
        or isinstance(height, bool)
        or not isinstance(width, Integral)
        or not isinstance(height, Integral)
        or width <= 0
        or height <= 0
    ):
        raise ValueError("frame_size width and height must be positive integers")
    return int(height), int(width)


def _resolve_frame_id(frame_id: str | int | None) -> str:
    if frame_id is None:
        return f"auto-{next(_FRAME_ID_COUNTER)}"
    if isinstance(frame_id, bool):
        raise ValueError("frame_id must be a non-empty string or non-negative integer")
    if isinstance(frame_id, Integral) and frame_id >= 0:
        return str(int(frame_id))
    if isinstance(frame_id, str) and frame_id.strip():
        return frame_id.strip()
    raise ValueError("frame_id must be a non-empty string or non-negative integer")


def _tracking_object_id(track_id: Any) -> str:
    if isinstance(track_id, str) and track_id.startswith("track:"):
        return track_id
    return f"track:{track_id}"


def bounding_box_iou(
    first_box: Sequence[float],
    second_box: Sequence[float],
) -> float:
    """Return intersection over union for two xyxy boxes."""
    first = _coerce_box(first_box)
    second = _coerce_box(second_box)
    if first is None or second is None:
        return 0.0

    left = max(first[0], second[0])
    top = max(first[1], second[1])
    right = min(first[2], second[2])
    bottom = min(first[3], second[3])
    intersection = max(0.0, right - left) * max(0.0, bottom - top)
    first_area = (first[2] - first[0]) * (first[3] - first[1])
    second_area = (second[2] - second[0]) * (second[3] - second[1])
    union = first_area + second_area - intersection
    return intersection / union if union > 0 else 0.0


def _coerce_box(value: Any) -> tuple[float, float, float, float] | None:
    try:
        coordinates = tuple(float(item) for item in value)
    except (TypeError, ValueError, OverflowError):
        return None
    if len(coordinates) != 4 or not all(np.isfinite(coordinates)):
        return None
    x1, y1, x2, y2 = coordinates
    if x1 >= x2 or y1 >= y2:
        return None
    return x1, y1, x2, y2


def _horizontal_position(center_x: float, width: int, center_zone_fraction: float) -> str:
    lower_boundary = width * (0.5 - center_zone_fraction)
    upper_boundary = width * (0.5 + center_zone_fraction)
    if center_x < lower_boundary:
        return "left"
    if center_x > upper_boundary:
        return "right"
    return "center"


def _vertical_position(center_y: float, height: int, center_zone_fraction: float) -> str:
    lower_boundary = height * (0.5 - center_zone_fraction)
    upper_boundary = height * (0.5 + center_zone_fraction)
    if center_y < lower_boundary:
        return "above"
    if center_y > upper_boundary:
        return "below"
    return "middle"


def _depth_value_for_box(
    depth_map: np.ndarray | None,
    box: tuple[float, float, float, float],
) -> float | None:
    if depth_map is None:
        return None
    height, width = depth_map.shape
    x1 = max(0, min(width, int(np.floor(box[0]))))
    y1 = max(0, min(height, int(np.floor(box[1]))))
    x2 = max(0, min(width, int(np.ceil(box[2]))))
    y2 = max(0, min(height, int(np.ceil(box[3]))))
    if x1 >= x2 or y1 >= y2:
        return None

    region = depth_map[y1:y2, x1:x2]
    finite_values = region[np.isfinite(region)]
    if finite_values.size == 0:
        return None
    return float(np.median(finite_values))


def calculate_spatial_relationships(
    detections: Iterable[Mapping[str, Any]] | None,
    image_shape: Sequence[int],
    depth_map: Any = None,
    *,
    frame_id: str | int | None = None,
    center_zone_fraction: float = 0.2,
    overlap_threshold: float = 0.1,
    depth_difference_threshold: float = DEFAULT_DEPTH_DIFFERENCE_THRESHOLD,
    higher_depth_is_nearer: bool = True,
) -> dict[str, Any]:
    """Calculate 2D box relations and optional pairwise relative-depth relations.

    ``image_shape`` is canonically ``(height, width)``; convert external
    ``{width, height}`` with ``image_shape_from_frame_size``. Boxes use pixel
    ``xyxy`` coordinates. Untracked IDs use ``temporary:<frame-id>:<index>``.
    Higher normalized Depth Anything values are treated as nearer by default.
    Depth values are medians of finite map pixels inside clipped box regions.
    """
    resolved_frame_id = _resolve_frame_id(frame_id)
    if not 0 <= center_zone_fraction < 0.5:
        raise ValueError("center_zone_fraction must be between 0 and 0.5")
    if not 0 < overlap_threshold <= 1:
        raise ValueError("overlap_threshold must be greater than 0 and at most 1")
    if not np.isfinite(depth_difference_threshold) or depth_difference_threshold < 0:
        raise ValueError("depth_difference_threshold must be finite and non-negative")

    warnings: list[str] = []
    try:
        height, width = (int(value) for value in image_shape)
    except (TypeError, ValueError, OverflowError):
        height, width = 0, 0
    if height <= 0 or width <= 0:
        return {
            "frame_id": resolved_frame_id,
            "image_shape": (height, width),
            "objects": [],
            "relationships": [],
            "warnings": ["image_shape must contain positive (height, width) dimensions"],
        }

    valid_depth: np.ndarray | None = None
    if depth_map is not None:
        try:
            candidate_depth = np.asarray(depth_map, dtype=np.float32)
        except (TypeError, ValueError, OverflowError):
            candidate_depth = np.empty((0, 0), dtype=np.float32)
        if candidate_depth.ndim != 2:
            warnings.append("depth_map must be a 2D array; relative-depth comparisons were skipped")
        elif candidate_depth.shape != (height, width):
            warnings.append(
                f"depth_map dimensions {candidate_depth.shape} do not match image_shape "
                f"({height}, {width}); relative-depth comparisons were skipped"
            )
        elif not np.isfinite(candidate_depth).any():
            warnings.append("depth_map contains no finite values; relative-depth comparisons were skipped")
        else:
            valid_depth = candidate_depth

    objects: list[dict[str, Any]] = []
    if detections is None:
        warnings.append("detections were not supplied")
        detection_items: list[Mapping[str, Any]] = []
    else:
        try:
            detection_items = list(detections)
        except TypeError:
            detection_items = []
            warnings.append("detections must be an iterable of mappings")

    for index, detection in enumerate(detection_items):
        if not isinstance(detection, Mapping):
            warnings.append(f"detection {index} is not a mapping and was skipped")
            continue

        raw_box = detection.get("bounding_box", detection.get("bbox"))
        box = _coerce_box(raw_box)
        if box is None:
            warnings.append(f"detection {index} has an invalid xyxy bounding box and was skipped")
            continue

        clipped_box = (
            max(0.0, min(float(width), box[0])),
            max(0.0, min(float(height), box[1])),
            max(0.0, min(float(width), box[2])),
            max(0.0, min(float(height), box[3])),
        )
        if clipped_box[0] >= clipped_box[2] or clipped_box[1] >= clipped_box[3]:
            warnings.append(f"detection {index} bounding box lies outside the image and was skipped")
            continue

        track_id = detection.get("track_id", detection.get("id"))
        if hasattr(track_id, "item"):
            track_id = track_id.item()
        object_id = _tracking_object_id(track_id) if track_id is not None else f"temporary:{resolved_frame_id}:{index}"
        class_name = detection.get("class_name", detection.get("class", detection.get("name", "object")))
        center_x = (clipped_box[0] + clipped_box[2]) / 2
        center_y = (clipped_box[1] + clipped_box[3]) / 2
        object_depth = _depth_value_for_box(valid_depth, clipped_box)

        spatial_object = {
            "object_id": object_id,
            "track_id": track_id,
            "class_name": str(class_name),
            "bounding_box": clipped_box,
            "center": (center_x, center_y),
            "horizontal_position": _horizontal_position(center_x, width, center_zone_fraction),
            "vertical_position": _vertical_position(center_y, height, center_zone_fraction),
            "relative_depth_value": object_depth,
        }
        if "confidence" in detection and detection["confidence"] is not None:
            confidence = detection["confidence"]
            if hasattr(confidence, "item"):
                confidence = confidence.item()
            try:
                confidence = float(confidence)
            except (TypeError, ValueError, OverflowError):
                warnings.append(f"detection {index} has invalid confidence and it was omitted")
            else:
                if np.isfinite(confidence):
                    spatial_object["confidence"] = confidence
                else:
                    warnings.append(f"detection {index} has non-finite confidence and it was omitted")
        objects.append(spatial_object)

    relationships: list[dict[str, Any]] = []
    for first_index, first in enumerate(objects):
        first_box = first["bounding_box"]
        first_center_x, first_center_y = first["center"]
        for second in objects[first_index + 1:]:
            if first["object_id"] == second["object_id"]:
                warnings.append(f"duplicate object ID {first['object_id']!r}; skipped self-relationships")
                continue

            second_box = second["bounding_box"]
            second_center_x, second_center_y = second["center"]

            if first_center_x < second_center_x:
                relationships.append({"subject": first["object_id"], "relation": "left_of", "object": second["object_id"]})
                relationships.append({"subject": second["object_id"], "relation": "right_of", "object": first["object_id"]})
            elif first_center_x > second_center_x:
                relationships.append({"subject": first["object_id"], "relation": "right_of", "object": second["object_id"]})
                relationships.append({"subject": second["object_id"], "relation": "left_of", "object": first["object_id"]})
            if first_center_y < second_center_y:
                relationships.append({"subject": first["object_id"], "relation": "above", "object": second["object_id"]})
                relationships.append({"subject": second["object_id"], "relation": "below", "object": first["object_id"]})
            elif first_center_y > second_center_y:
                relationships.append({"subject": first["object_id"], "relation": "below", "object": second["object_id"]})
                relationships.append({"subject": second["object_id"], "relation": "above", "object": first["object_id"]})

            overlap = bounding_box_iou(first_box, second_box)
            if overlap >= overlap_threshold:
                relationships.append(
                    {
                        "subject": first["object_id"],
                        "relation": "overlapping",
                        "object": second["object_id"],
                        "iou": overlap,
                    }
                )
                relationships.append(
                    {
                        "subject": second["object_id"],
                        "relation": "overlapping",
                        "object": first["object_id"],
                        "iou": overlap,
                    }
                )

            first_depth = first["relative_depth_value"]
            second_depth = second["relative_depth_value"]
            if first_depth is not None and second_depth is not None:
                difference = first_depth - second_depth
                if abs(difference) <= depth_difference_threshold:
                    relationships.append(
                        {
                            "subject": first["object_id"],
                            "relation": "similar_depth",
                            "object": second["object_id"],
                            "depth_difference": abs(difference),
                            "depth_difference_threshold": depth_difference_threshold,
                        }
                    )
                    relationships.append(
                        {
                            "subject": second["object_id"],
                            "relation": "similar_depth",
                            "object": first["object_id"],
                            "depth_difference": abs(difference),
                            "depth_difference_threshold": depth_difference_threshold,
                        }
                    )
                else:
                    first_is_nearer = difference > 0 if higher_depth_is_nearer else difference < 0
                    first_rel = "nearer_than" if first_is_nearer else "farther_than"
                    second_rel = "farther_than" if first_is_nearer else "nearer_than"
                    relationships.append(
                        {
                            "subject": first["object_id"],
                            "relation": first_rel,
                            "object": second["object_id"],
                            "depth_difference": abs(difference),
                            "depth_difference_threshold": depth_difference_threshold,
                        }
                    )
                    relationships.append(
                        {
                            "subject": second["object_id"],
                            "relation": second_rel,
                            "object": first["object_id"],
                            "depth_difference": abs(difference),
                            "depth_difference_threshold": depth_difference_threshold,
                        }
                    )

    return {
        "frame_id": resolved_frame_id,
        "image_shape": (height, width),
        "objects": objects,
        "relationships": relationships,
        "warnings": warnings,
    }