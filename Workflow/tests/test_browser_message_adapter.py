from Backend.browser_message_adapter import (
    adapt_browser_frame_result,
    adapt_browser_query_result,
    dashboard_reasoning,
)


def test_frame_adapter_echoes_browser_id_and_maps_only_real_UI_fields():
    result = {
        "type": "frame_result",
        "frame_id": "browser:session:7",
        "frame_number": 12,
        "frame_size": {"width": 640, "height": 480},
        "status": "success",
        "detections": [
            {"track_id": 8, "class_name": "bottle", "confidence": 0.87, "bounding_box": [1, 2, 3, 4]},
            {"class_name": "chair", "confidence": 0.7, "bounding_box": [5, 6, 7, 8]},
        ],
        "objects": [{"object_id": "track:8", "class_name": "bottle"}],
        "relationships": [{"subject": "track:8", "relation": "left_of", "object": "track:9"}],
        "scene_summary_speech": {"spoken_text": "A bottle is visible.", "source_status": "visible"},
        "stage_status": {"detection": "ready"},
        "depth_preview": "actual-depth-preview",
    }

    adapted = adapt_browser_frame_result(result, browser_frame_id=7)

    assert adapted["frame_id"] == 7
    assert adapted["frame_number"] == 12
    assert adapted["tracks"] == [result["detections"][0]]
    assert adapted["relationships"][0]["source"] == "track:8"
    assert adapted["relationships"][0]["target"] == "track:9"
    assert adapted["relationships"][0]["relationship"] == "left_of"
    assert adapted["scene_narration"] == "A bottle is visible."
    assert adapted["stage_status"]["detection"] == "running"
    assert "processing_ms" not in adapted
    assert "events" not in adapted
    assert adapted["depth_preview"] == "actual-depth-preview"


def test_frame_adapter_passes_through_measured_processing_ms():
    adapted = adapt_browser_frame_result(
        {"type": "frame_result", "processing_ms": 123.4, "stage_status": {"detection": "ready"}},
        browser_frame_id=1,
    )
    assert adapted["processing_ms"] == 123.4
    assert adapted["stage_status"]["detection"] == "running"


def test_adapter_omits_UI_fields_without_actual_source_data():
    adapted = adapt_browser_frame_result(
        {"type": "frame_result", "detections": None, "relationships": None},
        browser_frame_id=3,
    )

    assert adapted["frame_id"] == 3
    assert "tracks" not in adapted
    assert "scene_narration" not in adapted
    assert "processing_fps" not in adapted
    assert "depth_preview" not in adapted


def test_query_adapter_aliases_actual_answer_objects_and_spoken_text():
    query_result = {
        "type": "query_result",
        "query": "Where is the bottle?",
        "status": "visible",
        "answer": "A bottle is on the left.",
        "matched_objects": [{"class_name": "bottle", "horizontal_position": "left"}],
        "evidence": {"object_state": {"status": "visible"}},
        "warnings": [],
        "speech_response": {"spoken_text": "A bottle is on the left.", "source_status": "visible"},
    }

    adapted = adapt_browser_query_result(query_result)

    assert adapted["narration"] == query_result["speech_response"]["spoken_text"]
    assert adapted["reasoning"]["objects"] == query_result["matched_objects"]
    assert adapted["reasoning"]["evidence"] == query_result["evidence"]
    assert "audio" not in adapted


def test_dashboard_reasoning_is_none_until_an_actual_query_exists():
    assert dashboard_reasoning(None) is None