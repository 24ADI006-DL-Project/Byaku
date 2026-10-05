from collections.abc import Mapping
import re
from typing import Any


def _collect_warnings(source: Mapping[str, Any], warnings: list[str]) -> list[str]:
    raw_warnings = source.get("warnings", [])
    if not isinstance(raw_warnings, (list, tuple)):
        warnings.append("warnings field must be a list or tuple")
        return []

    collected = []
    for warning in raw_warnings:
        if isinstance(warning, str) and warning.strip():
            value = re.sub(r"\s+", " ", warning.strip())
            if value not in collected:
                collected.append(value)
        elif warning is not None:
            warnings.append("a malformed warning entry was omitted")
    return collected


def _spoken_sentences(value: Any, warnings: list[str]) -> str:
    if not isinstance(value, str):
        return ""
    normalized = re.sub(r"\s+", " ", value).strip()
    if not normalized:
        return ""

    warning_keys = {warning.casefold().rstrip(".!?") for warning in warnings}
    sentences = re.split(r"(?<=[.!?])\s+", normalized)
    spoken = []
    seen = set()
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
        folded = sentence.casefold().rstrip(".!?")
        if re.match(r"^(?:warning|warnings|debug note|internal note)\s*:", sentence, flags=re.IGNORECASE):
            continue
        if folded in warning_keys or folded in seen:
            continue
        seen.add(folded)
        spoken.append(sentence)
        if len(spoken) == 2:
            break
    return " ".join(spoken)


def _class_name(result: Mapping[str, Any]) -> str | None:
    objects = result.get("matched_objects", [])
    if not isinstance(objects, (list, tuple)):
        return None
    for item in objects:
        if isinstance(item, Mapping):
            class_name = item.get("class_name")
            if isinstance(class_name, str) and class_name.strip():
                return class_name.strip()
    return None


def _article(class_name: str) -> str:
    first_letter = next((character.casefold() for character in class_name if character.isalpha()), "")
    return f"{'an' if first_letter in {'a', 'e', 'i', 'o', 'u'} else 'a'} {class_name}"


def _structured_response(spoken_text: str, source_status: str, warnings: list[str]) -> dict[str, Any]:
    return {
        "spoken_text": spoken_text,
        "source_status": source_status,
        "warnings": warnings,
    }


def prepare_spoken_response(query_result: Any) -> dict[str, Any]:
    """Prepare concise spoken text from a query-reasoning result without TTS."""
    if not isinstance(query_result, Mapping):
        return _structured_response(
            "I couldn't prepare a response because the query result is malformed.",
            "malformed",
            ["query result must be a mapping"],
        )

    warnings: list[str] = []
    source_warnings = _collect_warnings(query_result, warnings)
    status_value = query_result.get("status")
    status = status_value if isinstance(status_value, str) and status_value else "unknown"
    class_name = _class_name(query_result)

    if status in {"visible", "answered"}:
        spoken_text = _spoken_sentences(query_result.get("answer"), source_warnings)
        if not spoken_text:
            spoken_text = f"{_article(class_name).capitalize()} is currently visible." if class_name else "An object is currently visible."
    elif status == "missing":
        spoken_text = (
            f"{_article(class_name).capitalize()} is not currently visible; it may be temporarily missing."
            if class_name
            else "The object is not currently visible; it may be temporarily missing."
        )
    elif status == "stale":
        spoken_text = (
            f"{_article(class_name).capitalize()}'s last-known location may no longer be reliable."
            if class_name
            else "The last-known location may no longer be reliable."
        )
    elif status == "not_found":
        spoken_text = "The requested object was not found in the available scene data."
    elif status == "ambiguous":
        spoken_text = "Multiple matching objects were found. Please specify which one you mean."
    elif status == "insufficient_evidence":
        spoken_text = "The available information is insufficient to answer that reliably."
    elif status == "empty_scene":
        spoken_text = "No objects are currently visible."
    elif status == "unsupported":
        spoken_text = "I can answer questions about visible objects, supported locations, and last-seen information."
    elif status == "not_ready":
        spoken_text = "No scene has been processed yet. Please wait for the camera."
    else:
        warnings.append(f"unknown query status {status_value!r}")
        status = "unknown"
        spoken_text = "I can't prepare a reliable spoken response because the result status is unknown."

    return _structured_response(spoken_text, status, warnings + source_warnings)


def prepare_spoken_scene_summary(scene_result: Any) -> dict[str, Any]:
    """Prepare concise spoken text from the scene-description engine's output."""
    if not isinstance(scene_result, Mapping):
        return _structured_response(
            "I couldn't prepare a scene summary because the input is malformed.",
            "malformed",
            ["scene description must be a mapping"],
        )

    warnings: list[str] = []
    source_warnings = _collect_warnings(scene_result, warnings)
    raw_count = scene_result.get("visible_object_count")
    count = raw_count if isinstance(raw_count, int) and not isinstance(raw_count, bool) and raw_count >= 0 else None
    if raw_count is not None and count is None:
        warnings.append("visible_object_count must be a non-negative integer")

    if count == 0:
        spoken_text = "No objects are currently visible."
        source_status = "empty_scene"
    else:
        description = _spoken_sentences(scene_result.get("description"), source_warnings)
        says_empty = bool(re.search(r"\bno objects are currently visible\b", description, flags=re.IGNORECASE))
        if count is not None and count > 0 and says_empty:
            warnings.append("description conflicts with visible_object_count; used the count")
            description = ""
        if description:
            spoken_text = description
            source_status = "visible" if count is not None else "unknown"
            if count is None:
                warnings.append("visible_object_count is unavailable; scene visibility could not be verified")
        elif count is not None and count > 0:
            spoken_text = f"There {'is' if count == 1 else 'are'} {count} visible object{'s' if count != 1 else ''}."
            source_status = "visible"
        else:
            spoken_text = "A reliable scene summary is not available."
            source_status = "insufficient_evidence"

    return _structured_response(spoken_text, source_status, warnings + source_warnings)