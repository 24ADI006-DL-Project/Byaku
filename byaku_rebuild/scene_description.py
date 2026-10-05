from collections.abc import Mapping, Sequence
import math
from typing import Any

import networkx as nx

from .scene_graph import get_objects
from .spatial_relationships import bounding_box_iou


RELATION_PHRASES = {
    "left_of": "to the left of",
    "right_of": "to the right of",
    "above": "above",
    "below": "below",
    "nearer_than": "in front of",
    "farther_than": "behind",
    "similar_depth": "at a similar relative depth to",
}
VALID_RELATION_TYPES = {*RELATION_PHRASES, "overlapping"}
INVERSE_RELATIONS = {
    "left_of": "right_of",
    "right_of": "left_of",
    "above": "below",
    "below": "above",
    "nearer_than": "farther_than",
    "farther_than": "nearer_than",
}
SYMMETRIC_RELATIONS = {"overlapping", "similar_depth"}


def _as_records(value: Any, field_name: str, warnings: list[str]) -> list[Mapping[str, Any]]:
    if not isinstance(value, (list, tuple)):
        warnings.append(f"{field_name} must be a list or tuple; ignored it")
        return []

    records = []
    for index, item in enumerate(value):
        if isinstance(item, Mapping):
            records.append(item)
        else:
            warnings.append(f"{field_name} entry {index} is malformed and was ignored")
    return records


def _valid_id(value: Any) -> bool:
    if isinstance(value, bool) or value is None:
        return False
    if isinstance(value, (str, int)):
        return bool(value.strip()) if isinstance(value, str) else True
    if isinstance(value, float):
        return math.isfinite(value)
    return False


def _valid_box(value: Any) -> bool:
    try:
        box = tuple(float(part) for part in value)
    except (TypeError, ValueError, OverflowError):
        return False
    return (
        len(box) == 4
        and all(math.isfinite(part) for part in box)
        and box[0] < box[2]
        and box[1] < box[3]
    )


def _valid_center(value: Any) -> bool:
    try:
        center = tuple(float(part) for part in value)
    except (TypeError, ValueError, OverflowError):
        return False
    return len(center) == 2 and all(math.isfinite(part) for part in center)


def _valid_depth(value: Any) -> bool:
    try:
        depth = float(value)
    except (TypeError, ValueError, OverflowError):
        return False
    return math.isfinite(depth)


def _object_id(item: Mapping[str, Any], index: int) -> Any:
    object_id = item.get("object_id")
    if object_id is not None:
        return object_id
    track_id = item.get("track_id")
    if track_id is not None and _valid_id(track_id):
        return f"track:{track_id}"
    return f"frame:{index}"


def _visible_objects(
    spatial_output: Mapping[str, Any] | None,
    scene_graph: nx.MultiDiGraph | None,
    scene_memory: Any,
    warnings: list[str],
) -> list[dict[str, Any]]:
    if scene_memory is not None:
        if isinstance(scene_memory, Mapping):
            visible = scene_memory.get("visible_objects", [])
            missing = scene_memory.get("temporarily_missing_objects", scene_memory.get("missing_objects", []))
            stale = scene_memory.get("stale_objects", [])
            memory_warnings = scene_memory.get("warnings", [])
        elif callable(getattr(scene_memory, "get_visible_objects", None)):
            try:
                visible = scene_memory.get_visible_objects()
                missing = scene_memory.get_temporarily_missing_objects()
                stale = scene_memory.get_stale_objects()
                memory_warnings = []
            except Exception as error:
                warnings.append(f"scene memory query failed: {type(error).__name__}: {error}")
                return []
        else:
            warnings.append("scene_memory must be a SceneMemory instance or update snapshot; no objects described")
            return []

        if isinstance(memory_warnings, (list, tuple)):
            warnings.extend(str(message) for message in memory_warnings)
        missing_count = len(missing) if isinstance(missing, (list, tuple)) else 0
        stale_count = len(stale) if isinstance(stale, (list, tuple)) else 0
        if missing_count or stale_count:
            warnings.append(
                f"Excluded {missing_count} temporarily missing and {stale_count} stale objects from the current description"
            )
        if not isinstance(visible, (list, tuple)):
            warnings.append("scene memory visible_objects must be a list or tuple")
            return []
        candidates = visible
    elif isinstance(scene_graph, nx.MultiDiGraph):
        candidates = get_objects(scene_graph)
    elif spatial_output is None:
        candidates = []
    elif isinstance(spatial_output, Mapping):
        candidates = spatial_output.get("objects", [])
    else:
        warnings.append("spatial_output must be a mapping")
        return []

    if not isinstance(candidates, (list, tuple)):
        warnings.append("visible objects must be a list or tuple")
        return []

    visible_objects: list[dict[str, Any]] = []
    seen_ids: set[Any] = set()
    for index, item in enumerate(candidates):
        if not isinstance(item, Mapping):
            warnings.append(f"object {index} is malformed and was ignored")
            continue
        status = item.get("status")
        if status is not None and status != "visible":
            warnings.append(f"object {index} has status {status!r} and was excluded from the current description")
            continue
        class_name = item.get("class_name")
        if not isinstance(class_name, str) or not class_name.strip():
            warnings.append(f"object {index} has no valid class name and was ignored")
            continue
        object_id = _object_id(item, index)
        if not _valid_id(object_id):
            warnings.append(f"object {index} has an invalid object ID and was ignored")
            continue
        try:
            hash(object_id)
        except TypeError:
            warnings.append(f"object {index} has an unhashable object ID and was ignored")
            continue
        if object_id in seen_ids:
            warnings.append(f"duplicate visible object ID {object_id!r}; retained the first record")
            continue
        seen_ids.add(object_id)
        record = dict(item)
        record["object_id"] = object_id
        record["class_name"] = class_name.strip()
        visible_objects.append(record)
    return visible_objects


def _input_relationships(
    spatial_output: Mapping[str, Any] | None,
    scene_graph: nx.MultiDiGraph | None,
    warnings: list[str],
) -> list[Mapping[str, Any]]:
    if isinstance(scene_graph, nx.MultiDiGraph):
        return [
            {
                "subject": source,
                "object": destination,
                "relation": attributes.get("relation", key),
                **dict(attributes),
            }
            for source, destination, key, attributes in scene_graph.edges(keys=True, data=True)
        ]
    if scene_graph is not None:
        warnings.append("scene_graph must be a NetworkX MultiDiGraph; using spatial relationships if available")
    if spatial_output is None:
        return []
    if not isinstance(spatial_output, Mapping):
        return []
    return _as_records(spatial_output.get("relationships", []), "relationships", warnings)


def _has_relationship_evidence(
    relation: Mapping[str, Any],
    subject: Mapping[str, Any],
    target: Mapping[str, Any],
) -> bool:
    relation_type = relation["relation"]

    def center(item: Mapping[str, Any]) -> tuple[float, float] | None:
        value = item.get("center")
        if _valid_center(value):
            return float(value[0]), float(value[1])
        box = item.get("bounding_box", item.get("bbox"))
        if _valid_box(box):
            coordinates = tuple(float(part) for part in box)
            return (coordinates[0] + coordinates[2]) / 2, (coordinates[1] + coordinates[3]) / 2
        return None

    if relation_type in {"left_of", "right_of", "above", "below"}:
        subject_center = center(subject)
        target_center = center(target)
        if subject_center is None or target_center is None:
            return False
        if relation_type == "left_of":
            return subject_center[0] < target_center[0]
        if relation_type == "right_of":
            return subject_center[0] > target_center[0]
        if relation_type == "above":
            return subject_center[1] < target_center[1]
        return subject_center[1] > target_center[1]
    if relation_type == "overlapping":
        try:
            iou = float(relation.get("iou"))
        except (TypeError, ValueError, OverflowError):
            return False
        subject_box = subject.get("bounding_box", subject.get("bbox"))
        target_box = target.get("bounding_box", target.get("bbox"))
        return (
            _valid_box(subject_box)
            and _valid_box(target_box)
            and math.isfinite(iou)
            and iou > 0
            and bounding_box_iou(subject_box, target_box) > 0
        )
    if relation_type in {"nearer_than", "farther_than", "similar_depth"}:
        return _valid_depth(subject.get("relative_depth_value")) and _valid_depth(
            target.get("relative_depth_value")
        )
    return False


def _deduplicate_relationships(
    relationships: list[Mapping[str, Any]],
    objects_by_id: Mapping[Any, Mapping[str, Any]],
    object_order: Mapping[Any, int],
    warnings: list[str],
) -> list[dict[str, Any]]:
    accepted: list[dict[str, Any]] = []
    seen_relationships: dict[tuple[Any, str, Any], tuple[Any, str, Any]] = {}
    for index, item in enumerate(relationships):
        source = item.get("subject")
        destination = item.get("object")
        relation_type = item.get("relation")
        if not _valid_id(source) or not _valid_id(destination) or not isinstance(relation_type, str):
            warnings.append(f"relationship {index} is malformed and was ignored")
            continue
        if relation_type not in VALID_RELATION_TYPES:
            warnings.append(f"relationship {index} has unsupported type {relation_type!r} and was ignored")
            continue
        if source == destination or source not in objects_by_id or destination not in objects_by_id:
            warnings.append(f"relationship {index} references a non-visible or invalid object and was ignored")
            continue
        subject = objects_by_id[source]
        target = objects_by_id[destination]
        if not _has_relationship_evidence(item, subject, target):
            warnings.append(f"relationship {index} lacks required geometry or relative-depth evidence and was ignored")
            continue

        canonical_source, canonical_target, canonical_relation = source, destination, relation_type
        if relation_type in INVERSE_RELATIONS and object_order[source] > object_order[destination]:
            canonical_source, canonical_target = destination, source
            canonical_relation = INVERSE_RELATIONS[relation_type]
        elif relation_type in SYMMETRIC_RELATIONS and object_order[source] > object_order[destination]:
            canonical_source, canonical_target = destination, source
        signature = (canonical_source, canonical_relation, canonical_target)
        if signature in seen_relationships:
            previous_source, previous_relation, previous_target = seen_relationships[signature]
            is_reciprocal = (
                source == previous_target
                and destination == previous_source
                and (
                    INVERSE_RELATIONS.get(previous_relation) == relation_type
                    or (previous_relation == relation_type and relation_type in SYMMETRIC_RELATIONS)
                )
            )
            if not is_reciprocal:
                warnings.append(f"duplicate {relation_type!r} relationship was ignored")
            continue
        seen_relationships[signature] = (source, relation_type, destination)
        accepted.append(dict(item))
    return accepted


def _article(class_name: str) -> str:
    first_letter = next((character.lower() for character in class_name if character.isalpha()), "")
    article = "an" if first_letter in {"a", "e", "i", "o", "u"} else "a"
    return f"{article} {class_name}"


def _position_phrase(item: Mapping[str, Any]) -> str:
    horizontal = item.get("horizontal_position")
    vertical = item.get("vertical_position")
    if horizontal not in {"left", "center", "right"}:
        horizontal = None
    if vertical not in {"above", "middle", "below"}:
        vertical = None

    if vertical in {"above", "below"} and horizontal is not None:
        vertical_word = "upper" if vertical == "above" else "lower"
        horizontal_word = "center" if horizontal == "center" else horizontal
        return f"in the {vertical_word} {horizontal_word}"
    if horizontal == "left":
        return "on the left"
    if horizontal == "right":
        return "on the right"
    if horizontal == "center":
        return "near the center"
    if vertical == "above":
        return "in the upper part of the image"
    if vertical == "below":
        return "in the lower part of the image"
    return ""


def _relationship_sentence(
    relation: Mapping[str, Any],
    objects_by_id: Mapping[Any, Mapping[str, Any]],
) -> str:
    subject = objects_by_id[relation["subject"]]
    target = objects_by_id[relation["object"]]
    subject_name = _article(subject["class_name"])
    target_name = _article(target["class_name"])
    relation_type = relation["relation"]
    if relation_type == "overlapping":
        return f"{subject_name.capitalize()} overlaps {target_name}."
    phrase = RELATION_PHRASES[relation_type]
    return f"{subject_name.capitalize()} is {phrase} {target_name}."


def describe_scene(
    spatial_output: Mapping[str, Any] | None = None,
    *,
    scene_graph: nx.MultiDiGraph | None = None,
    scene_memory: Any = None,
) -> dict[str, Any]:
    """Return a concise deterministic description based only on supplied scene evidence.

    A supplied scene memory is authoritative for visibility. Without memory, the
    provided spatial output or graph nodes are treated as the current scene. Depth
    values are used only as relative evidence and are never converted to distances.
    """
    warnings: list[str] = []
    if isinstance(scene_graph, nx.MultiDiGraph):
        graph_warnings = scene_graph.graph.get("warnings", [])
        if isinstance(graph_warnings, (list, tuple)):
            warnings.extend(str(message) for message in graph_warnings)
    if isinstance(spatial_output, Mapping):
        upstream_warnings = spatial_output.get("warnings", [])
        if isinstance(upstream_warnings, (list, tuple)):
            warnings.extend(str(message) for message in upstream_warnings)

    visible_objects = _visible_objects(spatial_output, scene_graph, scene_memory, warnings)
    objects_by_id = {item["object_id"]: item for item in visible_objects}
    object_order = {item["object_id"]: index for index, item in enumerate(visible_objects)}
    input_relationships = _input_relationships(spatial_output, scene_graph, warnings)
    described_relationships = _deduplicate_relationships(
        input_relationships,
        objects_by_id,
        object_order,
        warnings,
    )

    sentences = [
        _relationship_sentence(relation, objects_by_id)
        for relation in described_relationships
    ]
    described_object_ids = {
        object_id
        for relation in described_relationships
        for object_id in (relation["subject"], relation["object"])
    }
    for item in visible_objects:
        if item["object_id"] in described_object_ids:
            continue
        position = _position_phrase(item)
        sentence = f"{_article(item['class_name']).capitalize()} is visible"
        if position:
            sentence += f" {position}"
        sentences.append(f"{sentence}.")

    description = " ".join(sentences) if sentences else "No objects are currently visible."
    return {
        "description": description,
        "visible_object_count": len(visible_objects),
        "described_relationships": described_relationships,
        "warnings": warnings,
    }