from collections.abc import Mapping
from typing import Any


def _records(value: Any) -> list[Mapping[str, Any]]:
    if not isinstance(value, (list, tuple)):
        return []
    return [item for item in value if isinstance(item, Mapping)]


def dashboard_reasoning(query_response: Mapping[str, Any] | None) -> dict[str, Any] | None:
    if query_response is None:
        return None
    return {
        "status": query_response.get("status", "unavailable"),
        "answer": query_response.get("answer"),
        "objects": list(query_response.get("matched_objects", [])),
        "evidence": query_response.get("evidence", {}),
        "warnings": list(query_response.get("warnings", [])),
    }


def adapt_browser_frame_result(
    frame_result: Mapping[str, Any],
    *,
    browser_frame_id: int,
) -> dict[str, Any]:
    """Map actual Step 19 frame fields to existing frontend aliases only."""
    response = dict(frame_result)
    response["frame_id"] = browser_frame_id

    detections = frame_result.get("detections")
    if isinstance(detections, (list, tuple)):
        response["tracks"] = [
            dict(detection)
            for detection in _records(detections)
            if detection.get("track_id") is not None
        ]

    relationships = frame_result.get("relationships")
    if isinstance(relationships, (list, tuple)):
        frontend_relationships = []
        for relationship in _records(relationships):
            adapted = dict(relationship)
            if "subject" in relationship:
                adapted["source"] = relationship["subject"]
            if "object" in relationship:
                adapted["target"] = relationship["object"]
            if "relation" in relationship:
                adapted["relationship"] = relationship["relation"]
            frontend_relationships.append(adapted)
        response["relationships"] = frontend_relationships

    summary = frame_result.get("scene_summary_speech")
    if isinstance(summary, Mapping) and isinstance(summary.get("spoken_text"), str):
        response["scene_narration"] = summary["spoken_text"]
    else:
        response.pop("scene_narration", None)

    # app.js line 112 checks stage_status?.detection === 'running' to show "AI processing".
    # The backend emits "ready" for a successful detection stage; map it to "running" here so
    # the indicator is correct. dashboard.js line 36 already normalises "ready"/"connected" to
    # "running" on its own, so this change is fully compatible with both pages.
    raw_stage = frame_result.get("stage_status")
    if isinstance(raw_stage, Mapping):
        mapped_stage = dict(raw_stage)
        for key in ("detection", "tracking"):
            if mapped_stage.get(key) in {"ready", "connected"}:
                mapped_stage[key] = "running"
        response["stage_status"] = mapped_stage

    return response


def adapt_browser_query_result(query_response: Mapping[str, Any]) -> dict[str, Any]:
    """Expose existing reasoning/speech values using dashboard/browser names."""
    response = dict(query_response)
    response["reasoning"] = dashboard_reasoning(query_response)
    speech = query_response.get("speech_response")
    if isinstance(speech, Mapping) and isinstance(speech.get("spoken_text"), str):
        response["narration"] = speech["spoken_text"]
    else:
        response.pop("narration", None)
    return response