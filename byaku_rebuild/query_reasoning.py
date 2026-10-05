from collections.abc import Mapping
import re
from typing import Any

import networkx as nx

from .scene_description import describe_scene
from .scene_graph import get_objects


RELATION_QUESTIONS = (
    (r"\bto\s+the\s+left\s+of\b|\bleft\s+of\b", "left_of"),
    (r"\bto\s+the\s+right\s+of\b|\bright\s+of\b", "right_of"),
    (r"\bin\s+front\s+of\b", "nearer_than"),
    (r"\bbehind\b", "farther_than"),
    (r"\babove\b", "above"),
    (r"\bbelow\b", "below"),
    (r"\boverlapping\b|\boverlaps\b", "overlapping"),
)
RELATION_TEXT = {
    "left_of": "to the left of",
    "right_of": "to the right of",
    "nearer_than": "in front of",
    "farther_than": "behind",
    "above": "above",
    "below": "below",
    "overlapping": "overlapping",
    "similar_depth": "at a similar relative depth to",
}
INVERSE_RELATION = {
    "left_of": "right_of",
    "right_of": "left_of",
    "above": "below",
    "below": "above",
    "nearer_than": "farther_than",
    "farther_than": "nearer_than",
}
OBJECT_FIELDS = (
    "object_id",
    "track_id",
    "class_name",
    "status",
    "horizontal_position",
    "vertical_position",
    "center",
    "bounding_box",
    "confidence",
    "relative_depth_value",
    "last_seen_frame",
    "last_seen_timestamp",
)


def _warning(warnings: list[str], message: str) -> None:
    if message not in warnings:
        warnings.append(message)


def _records(value: Any, field_name: str, warnings: list[str]) -> list[Mapping[str, Any]]:
    if not isinstance(value, (list, tuple)):
        _warning(warnings, f"{field_name} must be a list or tuple")
        return []
    output = []
    for index, item in enumerate(value):
        if isinstance(item, Mapping):
            output.append(item)
        else:
            _warning(warnings, f"{field_name} item {index} is malformed and was skipped")
    return output


def _object_id(item: Mapping[str, Any], index: int) -> Any:
    object_id = item.get("object_id")
    if object_id is not None:
        return object_id
    track_id = item.get("track_id")
    if track_id is not None:
        return f"track:{track_id}"
    return f"frame:{index}"


def _object_state(
    spatial_output: Mapping[str, Any] | None,
    scene_graph: nx.MultiDiGraph | None,
    scene_memory: Any,
    warnings: list[str],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], bool]:
    memory_available = scene_memory is not None
    if isinstance(scene_memory, Mapping):
        visible_raw = scene_memory.get("visible_objects", [])
        missing_raw = scene_memory.get("temporarily_missing_objects", scene_memory.get("missing_objects", []))
        stale_raw = scene_memory.get("stale_objects", [])
        memory_warnings = scene_memory.get("warnings", [])
    elif scene_memory is not None:
        try:
            visible_raw = scene_memory.get_visible_objects()
            missing_raw = scene_memory.get_temporarily_missing_objects()
            stale_raw = scene_memory.get_stale_objects()
            memory_warnings = []
        except Exception as error:
            _warning(warnings, f"scene memory query failed: {type(error).__name__}: {error}")
            return [], [], [], True
    elif isinstance(scene_graph, nx.MultiDiGraph):
        visible_raw = get_objects(scene_graph)
        missing_raw = []
        stale_raw = []
        memory_warnings = scene_graph.graph.get("warnings", [])
    elif isinstance(spatial_output, Mapping):
        visible_raw = spatial_output.get("objects", [])
        missing_raw = []
        stale_raw = []
        memory_warnings = spatial_output.get("warnings", [])
    else:
        visible_raw, missing_raw, stale_raw, memory_warnings = [], [], [], []
        if spatial_output is not None:
            _warning(warnings, "spatial_output must be a mapping")

    if isinstance(memory_warnings, (list, tuple)):
        for message in memory_warnings:
            _warning(warnings, str(message))

    visible_records = _records(visible_raw, "visible_objects", warnings)
    missing_records = _records(missing_raw, "temporarily_missing_objects", warnings)
    stale_records = _records(stale_raw, "stale_objects", warnings)
    spatial_objects: dict[Any, Mapping[str, Any]] = {}
    if isinstance(spatial_output, Mapping):
        for obj in spatial_output.get("objects", []):
            if isinstance(obj, Mapping) and "object_id" in obj:
                spatial_objects[obj["object_id"]] = obj

    buckets: dict[str, list[dict[str, Any]]] = {"visible": [], "missing": [], "stale": []}
    seen_ids: set[Any] = set()

    for default_status, records in (
        ("visible", visible_records),
        ("missing", missing_records),
        ("stale", stale_records),
    ):
        for index, item in enumerate(records):
            status = item.get("status", default_status)
            if status not in buckets:
                _warning(warnings, f"object has unsupported memory status {status!r} and was skipped")
                continue
            object_id = _object_id(item, index)
            try:
                hash(object_id)
            except TypeError:
                _warning(warnings, "object has an unhashable ID and was skipped")
                continue
            class_name = item.get("class_name")
            if not isinstance(class_name, str) or not class_name.strip():
                _warning(warnings, f"object {object_id!r} has no valid class name and was skipped")
                continue
            if object_id in seen_ids:
                _warning(warnings, f"duplicate object ID {object_id!r}; retained its first state")
                continue
            seen_ids.add(object_id)
            record = dict(item)
            record["object_id"] = object_id
            record["class_name"] = class_name.strip()
            record["status"] = status
            if object_id in spatial_objects:
                s_obj = spatial_objects[object_id]
                if record.get("horizontal_position") is None and "horizontal_position" in s_obj:
                    record["horizontal_position"] = s_obj["horizontal_position"]
                if record.get("vertical_position") is None and "vertical_position" in s_obj:
                    record["vertical_position"] = s_obj["vertical_position"]
            buckets[status].append(record)

    return buckets["visible"], buckets["missing"], buckets["stale"], memory_available


def _class_keys(records: list[dict[str, Any]]) -> list[str]:
    return sorted({item["class_name"].casefold() for item in records}, key=lambda name: (-len(name), name))


def _clean_noun_phrase(value: str) -> str:
    phrase = re.sub(r"[?.!,;:]+$", "", value.strip().casefold())
    phrase = re.sub(r"^(?:my|the|a|an|this|that)\s+", "", phrase)
    phrase = re.sub(r"\s+(?:currently\s+)?visible$", "", phrase)
    phrase = re.sub(r"\s+last\s+seen$", "", phrase)
    return phrase.strip()


def _resolve_classes(phrase: str, known_classes: list[str]) -> list[str]:
    normalized = _clean_noun_phrase(phrase)
    found = [
        class_name
        for class_name in known_classes
        if re.search(rf"(?<!\w){re.escape(class_name)}(?!\w)", normalized)
    ]
    if found:
        return found
    if normalized.endswith("s") and len(normalized) > 1:
        singular = normalized[:-1]
        if singular in known_classes:
            return [singular]
        normalized = singular
    return [normalized] if normalized else []


def _parse_query(query: str, known_classes: list[str]) -> dict[str, Any] | None:
    text = re.sub(r"\s+", " ", query.strip().casefold())
    # Expand common speech-recognition contractions and alternates.
    text = (
        text
        .replace("what's", "what is")
        .replace("where's", "where is")
        .replace("there's", "there is")
        .replace("i'm", "i am")
        .replace("can't", "cannot")
        .replace("don't", "do not")
    )
    if not text:
        return None

    # List-visible: "what objects are visible?", "what can you/i see?",
    # "what do you see?", "what is in front of me?", "what is around me?",
    # "describe the scene", "tell me what you see".
    if re.search(
        r"\b(?:what|which)\s+objects?\b.*\bvisible\b"
        r"|\bwhat\s+can\s+(?:i|you)\s+see\b"
        r"|\bwhat\s+(?:do|does)\s+(?:you|i)\s+see\b"
        r"|\bwhat\s+is\s+(?:in\s+front\s+of\s+me|around\s+me)\b"
        r"|\bdescribe\s+(?:the\s+)?scene\b"
        r"|\btell\s+me\s+what\s+you\s+see\b",
        text,
    ):
        return {"kind": "list_visible"}

    # Relational: "what is to the left/right/in front of/behind/above/below X?"
    for pattern, relation_type in RELATION_QUESTIONS:
        match = re.search(pattern, text)
        if match and re.search(r"^(?:what|which)\b", text):
            target_phrase = text[match.end():].strip()
            target_classes = _resolve_classes(target_phrase, known_classes)
            return {"kind": "relationship", "relation": relation_type, "target_classes": target_classes}

    # Last-seen: "where was/were X (last seen)?"
    last_seen_match = re.fullmatch(r"where\s+(?:was|were)\s+(.+?)(?:\s+last\s+seen)?[?.!]*", text)
    if last_seen_match:
        return {"kind": "last_seen", "classes": _resolve_classes(last_seen_match.group(1), known_classes)}
    if re.search(r"\bwhere\s+(?:was|were)\b", text):
        phrase = re.sub(r"^.*?\bwhere\s+(?:was|were)\s+", "", text)
        return {"kind": "last_seen", "classes": _resolve_classes(phrase, known_classes)}

    # Visibility check: "is/are X (currently) visible?"
    visible_match = re.fullmatch(
        r"(?:is|are)\s+(.+?)\s+(?:currently\s+)?visible[?.!]*",
        text,
    )
    if visible_match:
        return {"kind": "visibility", "classes": _resolve_classes(visible_match.group(1), known_classes)}

    # Nearby: "is there a X nearby?", "is there a X?", "can you see a X?",
    # "do you see a X?", "can i see a X?"
    nearby_match = re.fullmatch(
        r"(?:is\s+there\s+(?:a|an)\s+|can\s+(?:you|i)\s+see\s+(?:a|an)?\s*|do\s+you\s+see\s+(?:a|an)?\s*)(.+?)(?:\s+nearby|\s+around|\s+close)?[?.!]*",
        text,
    )
    if nearby_match:
        classes = _resolve_classes(nearby_match.group(1), known_classes)
        if classes:
            return {"kind": "visibility", "classes": classes}

    # "I cannot find my X" / "I can't find my X" / "I am looking for X"
    lost_match = re.search(
        r"\b(?:cannot|can not|do not|don't)\s+find\s+(?:my\s+|the\s+|a\s+)?(.+?)(?:\s+anywhere|\s+around\s+me)?[?.!]*$"
        r"|\bi\s+am\s+looking\s+for\s+(?:my\s+|the\s+|a\s+)?(.+?)(?:\s+anywhere|\s+around\s+me)?[?.!]*$"
        r"|\bwhere\s+(?:did\s+i\s+put|is)\s+(?:my\s+|the\s+|a\s+)?(.+?)[?.!]*$",
        text,
    )
    if lost_match:
        phrase = next(g for g in lost_match.groups() if g is not None)
        classes = _resolve_classes(phrase, known_classes)
        if classes:
            return {"kind": "location", "classes": classes}

    # Location: "where is/are X?"
    location_match = re.fullmatch(r"where\s+(?:is|are)\s+(.+?)[?.!]*", text)
    if location_match:
        return {"kind": "location", "classes": _resolve_classes(location_match.group(1), known_classes)}

    return None


def _matches_class(objects: list[dict[str, Any]], class_names: list[str]) -> list[dict[str, Any]]:
    selected = {class_name.casefold() for class_name in class_names}
    return [item for item in objects if item["class_name"].casefold() in selected]


def _public_object(item: Mapping[str, Any]) -> dict[str, Any]:
    return {field: item[field] for field in OBJECT_FIELDS if field in item}


def _result(
    query: str,
    answer: str,
    status: str,
    matched_objects: list[Mapping[str, Any]],
    evidence: Mapping[str, Any],
    warnings: list[str],
) -> dict[str, Any]:
    return {
        "query": query,
        "answer": answer,
        "status": status,
        "matched_objects": [_public_object(item) for item in matched_objects],
        "evidence": dict(evidence),
        "warnings": warnings,
    }


def _location_phrases(
    item: Mapping[str, Any],
    relationship_types: set[str] | None = None,
) -> list[str]:
    relationship_types = relationship_types or set()
    horizontal = item.get("horizontal_position")
    vertical = item.get("vertical_position")
    if horizontal not in {"left", "center", "right"}:
        horizontal = None
    if vertical not in {"above", "middle", "below"}:
        vertical = None

    horizontal_is_repeated = (horizontal == "left" and "left_of" in relationship_types) or (
        horizontal == "right" and "right_of" in relationship_types
    )
    vertical_is_repeated = (vertical == "above" and "above" in relationship_types) or (
        vertical == "below" and "below" in relationship_types
    )
    if horizontal_is_repeated:
        horizontal = None
    if vertical_is_repeated:
        vertical = None

    if vertical in {"above", "below"} and horizontal is not None:
        return [f"in the {'upper' if vertical == 'above' else 'lower'} {horizontal}"]
    phrases = []
    if horizontal == "left":
        phrases.append("in front of you, on the left")
    elif horizontal == "right":
        phrases.append("in front of you, on the right")
    elif horizontal == "center":
        phrases.append("in front of you, near the center")
    if vertical == "above":
        phrases.append("in the upper part of the image")
    elif vertical == "below":
        phrases.append("in the lower part of the image")
    return phrases


def _class_article(class_name: str) -> str:
    first = next((character.casefold() for character in class_name if character.isalpha()), "")
    return f"{'an' if first in {'a', 'e', 'i', 'o', 'u'} else 'a'} {class_name}"


def _state_answer(class_name: str, item: Mapping[str, Any], status: str) -> tuple[str, dict[str, Any]]:
    frame = item.get("last_seen_frame")
    timestamp = item.get("last_seen_timestamp")
    observations = item.get("observations")
    if isinstance(observations, (list, tuple)) and observations:
        latest = observations[-1]
        if isinstance(latest, Mapping):
            frame = latest.get("frame_number", frame)
            timestamp = latest.get("timestamp", timestamp)
    last_seen = f"frame {frame}" if frame is not None else f"timestamp {timestamp}" if timestamp is not None else None
    box = item.get("bounding_box")
    evidence = {
        "last_seen_frame": frame,
        "last_seen_timestamp": timestamp,
        "bounding_box": box,
    }
    if status == "missing":
        answer = f"{_class_article(class_name).capitalize()} is temporarily missing"
        if last_seen:
            answer += f"; it was last seen at {last_seen}"
        if box is not None:
            answer += f" (recorded box {box} pixels)"
        return answer + ".", evidence
    if status == "stale":
        answer = f"{_class_article(class_name).capitalize()}'s last location is stale and is not reliable"
        if last_seen:
            answer += f"; it was last seen at {last_seen}"
        if box is not None:
            answer += f" (last recorded box {box} pixels)"
        return answer + ".", evidence
    answer = f"{_class_article(class_name).capitalize()} is currently visible"
    if last_seen:
        answer += f"; its last observation was at {last_seen}"
    if box is not None:
        answer += f" (recorded box {box} pixels)"
    return answer + ".", evidence


def answer_query(
    query: Any,
    *,
    spatial_output: Mapping[str, Any] | None = None,
    scene_graph: nx.MultiDiGraph | None = None,
    scene_memory: Any = None,
    scene_age_seconds: float | None = None,
) -> dict[str, Any]:
    """Answer supported object/location queries using only supplied scene evidence.

    ``scene_age_seconds`` is the wall-clock age of the committed scene at query
    time. When it exceeds 30 seconds, answers include a staleness notice so the
    user is not misled by outdated scene data.

    Status values include ``visible``, ``missing``, ``stale``, ``not_found``,
    ``ambiguous``, ``insufficient_evidence``, ``unsupported``, and ``empty_scene``.
    """
    original_query = query if isinstance(query, str) else str(query)
    warnings: list[str] = []
    # Staleness flag: scenes older than 30 s are noted in the answer.
    _STALE_THRESHOLD_S = 30.0
    scene_is_stale = (
        isinstance(scene_age_seconds, (int, float))
        and not isinstance(scene_age_seconds, bool)
        and scene_age_seconds > _STALE_THRESHOLD_S
    )
    if scene_is_stale:
        _warning(warnings, f"scene data is {scene_age_seconds:.0f} seconds old and may not reflect the current view")
    if not isinstance(query, str) or not query.strip():
        return _result(
            original_query,
            "Please provide a non-empty question about visible objects or their locations.",
            "unsupported",
            [],
            {},
            ["query must be a non-empty string"],
        )

    try:
        scene_summary = describe_scene(
            spatial_output,
            scene_graph=scene_graph,
            scene_memory=scene_memory,
        )
    except Exception as error:
        scene_summary = {"description": "", "described_relationships": [], "warnings": []}
        _warning(warnings, f"scene description failed: {type(error).__name__}: {error}")

    visible, missing, stale, memory_available = _object_state(
        spatial_output,
        scene_graph,
        scene_memory,
        warnings,
    )
    for warning in scene_summary.get("warnings", []):
        _warning(warnings, str(warning))
    relationships = scene_summary.get("described_relationships", [])
    evidence: dict[str, Any] = {
        "scene_description": scene_summary.get("description", ""),
        "relationships": relationships,
    }

    if isinstance(spatial_output, Mapping) and "relationships" in spatial_output:
        relationships_available = isinstance(spatial_output.get("relationships"), (list, tuple))
    else:
        relationships_available = isinstance(scene_graph, nx.MultiDiGraph)

    parsed = _parse_query(query, _class_keys(visible + missing + stale))
    if parsed is None:
        return _result(
            original_query,
            "I can answer questions about visible objects, their supported spatial relationships, and remembered last-seen states.",
            "unsupported",
            [],
            evidence,
            warnings,
        )

    if parsed["kind"] == "list_visible":
        matched = visible
        status = "visible" if matched else "empty_scene"
        answer = scene_summary.get("description") or "No objects are currently visible."
        return _result(original_query, answer, status, matched, evidence, warnings)

    if parsed["kind"] == "relationship":
        classes = parsed["target_classes"]
        if not classes:
            return _result(original_query, "Which object's location should I use as the reference?", "unsupported", [], evidence, warnings)
        if len(classes) > 1:
            return _result(
                original_query,
                f"The reference could be {' or '.join(classes)}. Which object do you mean?",
                "ambiguous",
                [],
                evidence,
                warnings,
            )
        target_class = classes[0]
        targets = _matches_class(visible + missing + stale, [target_class])
        if len(targets) > 1:
            return _result(
                original_query,
                f"There are multiple {target_class} objects in the available scene memory. Which one do you mean?",
                "ambiguous",
                targets,
                evidence,
                warnings,
            )
        if not targets:
            return _result(
                original_query,
                f"I can't find a {target_class} in the available scene data.",
                "not_found",
                [],
                evidence,
                warnings,
            )
        target = targets[0]
        target_id = target["object_id"]
        if target["status"] != "visible":
            answer, state_evidence = _state_answer(target_class, target, target["status"])
            evidence["object_state"] = state_evidence
            return _result(original_query, answer, target["status"], [target], evidence, warnings)
        if not relationships_available:
            return _result(
                original_query,
                "I don't have spatial relationship data to answer that yet.",
                "insufficient_evidence",
                [target],
                evidence,
                warnings,
            )

        relation_type = parsed["relation"]
        visible_by_id = {item["object_id"]: item for item in visible}
        matches: list[dict[str, Any]] = []
        candidate_ids: list[Any] = []
        for relation in relationships:
            source = relation.get("subject")
            destination = relation.get("object")
            source_relation = relation.get("relation")
            if source_relation == relation_type and destination == target_id:
                candidate_id = source
                interpretation = "direct"
            elif (
                INVERSE_RELATION.get(source_relation) == relation_type
                and source == target_id
            ) or (
                source_relation == "overlapping"
                and relation_type == "overlapping"
                and source == target_id
            ):
                candidate_id = destination
                interpretation = "inverse"
            else:
                continue
            if candidate_id not in visible_by_id:
                continue
            if candidate_id not in candidate_ids:
                candidate_ids.append(candidate_id)
                matches.append(
                    {
                        **dict(relation),
                        "query_subject": candidate_id,
                        "query_relation": relation_type,
                        "query_object": target_id,
                        "interpretation": interpretation,
                    }
                )
        candidates = [visible_by_id[object_id] for object_id in candidate_ids]
        if len(candidates) > 1:
            names = ", ".join(_class_article(item["class_name"]) for item in candidates)
            answer = f"More than one object is {RELATION_TEXT[relation_type]} {_class_article(target_class)}: {names}."
            matched_objects = candidates + [target]
            status = "ambiguous"
        elif candidates:
            candidate = candidates[0]
            answer = f"{_class_article(candidate['class_name']).capitalize()} is {RELATION_TEXT[relation_type]} {_class_article(target_class)}."
            matched_objects = [candidate, target]
            status = "visible"
        else:
            answer = f"No visible object is recorded as {RELATION_TEXT[relation_type]} {_class_article(target_class)}."
            matched_objects = [target]
            status = "not_found"
            if not relationships:
                _warning(warnings, "no validated relationships were available")
        evidence["relationships"] = matches
        return _result(original_query, answer, status, matched_objects, evidence, warnings)

    classes = parsed.get("classes", [])
    if not classes:
        return _result(original_query, "Which object class do you mean?", "unsupported", [], evidence, warnings)
    if len(classes) > 1:
        return _result(
            original_query,
            f"The question mentions multiple object classes ({', '.join(classes)}). Which one do you mean?",
            "ambiguous",
            [],
            evidence,
            warnings,
        )

    class_name = classes[0]
    visible_matches = _matches_class(visible, [class_name])
    missing_matches = _matches_class(missing, [class_name])
    stale_matches = _matches_class(stale, [class_name])
    all_matches = visible_matches + missing_matches + stale_matches
    if len(all_matches) > 1:
        visible_count = len(visible_matches)
        missing_count = len(missing_matches)
        stale_count = len(stale_matches)
        answer = (
            f"There are multiple {class_name} instances in the available scene data "
            f"({visible_count} visible, {missing_count} temporarily missing, {stale_count} stale). "
            "Please clarify which one you mean."
        )
        return _result(original_query, answer, "ambiguous", all_matches, evidence, warnings)

    if not all_matches:
        return _result(
            original_query,
            f"I can't find a {class_name} in the available scene data.",
            "not_found",
            [],
            evidence,
            warnings,
        )

    item = all_matches[0]
    if item["status"] in {"missing", "stale"}:
        answer, state_evidence = _state_answer(class_name, item, item["status"])
        evidence["object_state"] = state_evidence
        return _result(original_query, answer, item["status"], [item], evidence, warnings)

    if parsed["kind"] == "visibility":
        return _result(
            original_query,
            f"Yes, {_class_article(class_name)} is currently visible.",
            "visible",
            [item],
            evidence,
            warnings,
        )

    if parsed["kind"] == "last_seen":
        if not memory_available and item.get("last_seen_frame") is None and item.get("last_seen_timestamp") is None:
            return _result(
                original_query,
                "Scene memory is not available, so I can't determine where it was last seen.",
                "insufficient_evidence",
                [item],
                evidence,
                warnings,
            )
        answer, state_evidence = _state_answer(class_name, item, "visible")
        evidence["object_state"] = state_evidence
        return _result(original_query, answer, "visible", [item], evidence, warnings)

    outgoing = [
        relation
        for relation in relationships
        if relation.get("subject") == item["object_id"]
    ]
    positions = _location_phrases(item, {relation.get("relation") for relation in outgoing})
    relation_phrases = []
    for relation in outgoing:
        target = next((obj for obj in visible if obj["object_id"] == relation.get("object")), None)
        if target is None:
            continue
        phrase = RELATION_TEXT.get(relation.get("relation"))
        if phrase is not None:
            relation_phrases.append(f"{phrase} {_class_article(target['class_name'])}")
    location_parts = list(dict.fromkeys(positions + relation_phrases))
    if location_parts:
        answer = f"{_class_article(class_name).capitalize()} is currently visible " + " and ".join(location_parts) + "."
        status = "visible"
    else:
        answer = f"{_class_article(class_name).capitalize()} is currently visible, but no supported location is available in the supplied data."
        status = "insufficient_evidence"
    if scene_is_stale:
        answer += " (Note: the scene data may be outdated.)"
    evidence["relationships"] = outgoing
    return _result(original_query, answer, status, [item], evidence, warnings)