from Workflow.reasoning.narration_response import (
    prepare_spoken_response,
    prepare_spoken_scene_summary,
)


def query_result(status: str, answer: str = "", **values: object) -> dict:
    result = {"query": "Where is the bottle?", "answer": answer, "status": status, "matched_objects": [], "evidence": {}, "warnings": []}
    result.update(values)
    return result


def test_visible_status_uses_supplied_answer_directly() -> None:
    result = prepare_spoken_response(
        query_result("visible", "A bottle is to the left of a chair.", matched_objects=[{"class_name": "bottle"}])
    )

    assert result["spoken_text"] == "A bottle is to the left of a chair."
    assert result["source_status"] == "visible"


def test_visible_status_without_answer_uses_only_matched_class() -> None:
    result = prepare_spoken_response(query_result("visible", matched_objects=[{"class_name": "apple"}]))

    assert result["spoken_text"] == "An apple is currently visible."
    assert "left" not in result["spoken_text"]
    assert "meters" not in result["spoken_text"]


def test_missing_status_says_not_currently_visible() -> None:
    result = prepare_spoken_response(
        query_result("missing", "The bottle is temporarily missing.", matched_objects=[{"class_name": "bottle"}])
    )

    assert result["spoken_text"] == "A bottle is not currently visible; it may be temporarily missing."
    assert "is currently visible" not in result["spoken_text"]
    assert result["source_status"] == "missing"


def test_stale_status_warns_last_known_location_may_be_unreliable() -> None:
    result = prepare_spoken_response(query_result("stale", matched_objects=[{"class_name": "chair"}]))

    assert result["spoken_text"] == "A chair's last-known location may no longer be reliable."
    assert "currently visible" not in result["spoken_text"]
    assert result["source_status"] == "stale"


def test_not_found_ambiguous_and_insufficient_statuses_are_clear() -> None:
    not_found = prepare_spoken_response(query_result("not_found"))
    ambiguous = prepare_spoken_response(query_result("ambiguous"))
    insufficient = prepare_spoken_response(query_result("insufficient_evidence"))

    assert "not found in the available scene data" in not_found["spoken_text"]
    assert "Multiple matching objects" in ambiguous["spoken_text"]
    assert "insufficient" in insufficient["spoken_text"]


def test_empty_scene_unsupported_and_unknown_statuses_are_safe() -> None:
    empty = prepare_spoken_response(query_result("empty_scene"))
    unsupported = prepare_spoken_response(query_result("unsupported"))
    unknown = prepare_spoken_response(query_result("future_status"))

    assert empty["spoken_text"] == "No objects are currently visible."
    assert unsupported["source_status"] == "unsupported"
    assert unknown["source_status"] == "unknown"
    assert unknown["warnings"] == ["unknown query status 'future_status'"]


def test_scene_summary_uses_description_and_visible_count() -> None:
    result = prepare_spoken_scene_summary(
        {
            "description": "A chair is to the left of a table.",
            "visible_object_count": 2,
            "described_relationships": [],
            "warnings": [],
        }
    )

    assert result["spoken_text"] == "A chair is to the left of a table."
    assert result["source_status"] == "visible"


def test_empty_scene_summary_does_not_repeat_stale_description() -> None:
    result = prepare_spoken_scene_summary(
        {"description": "A chair was last visible.", "visible_object_count": 0, "warnings": []}
    )

    assert result["spoken_text"] == "No objects are currently visible."
    assert result["source_status"] == "empty_scene"


def test_scene_summary_uses_count_when_description_is_missing_or_conflicts() -> None:
    missing_description = prepare_spoken_scene_summary({"visible_object_count": 3, "warnings": []})
    conflicting = prepare_spoken_scene_summary(
        {"description": "No objects are currently visible.", "visible_object_count": 2, "warnings": []}
    )

    assert missing_description["spoken_text"] == "There are 3 visible objects."
    assert conflicting["spoken_text"] == "There are 2 visible objects."
    assert conflicting["warnings"]


def test_duplicate_sentences_and_warning_text_are_removed_from_speech() -> None:
    result = prepare_spoken_response(
        query_result(
            "visible",
            "A bottle is visible. A bottle is visible. Warning: duplicate relationship ignored.",
            matched_objects=[{"class_name": "bottle"}],
            warnings=["duplicate relationship ignored"],
        )
    )

    assert result["spoken_text"] == "A bottle is visible."
    assert result["warnings"] == ["duplicate relationship ignored"]


def test_scene_summary_is_limited_to_two_sentences_and_deduplicated() -> None:
    result = prepare_spoken_scene_summary(
        {
            "description": "A chair is visible. A chair is visible. A table is nearby.",
            "visible_object_count": 2,
            "warnings": [],
        }
    )

    assert result["spoken_text"] == "A chair is visible. A table is nearby."
    assert result["spoken_text"].count(".") == 2


def test_malformed_results_missing_fields_and_bad_warning_values_are_safe() -> None:
    malformed_query = prepare_spoken_response(None)
    missing_status = prepare_spoken_response({"answer": "A bottle is visible."})
    malformed_scene = prepare_spoken_scene_summary([])
    bad_scene = prepare_spoken_scene_summary(
        {"description": "A bottle is visible.", "visible_object_count": "many", "warnings": "bad"}
    )

    assert malformed_query["source_status"] == "malformed"
    assert missing_status["source_status"] == "unknown"
    assert malformed_scene["source_status"] == "malformed"
    assert bad_scene["source_status"] == "unknown"
    assert bad_scene["warnings"]


def test_outputs_have_exact_structured_fields_and_are_deterministic() -> None:
    query = query_result("visible", "A bottle is visible.", matched_objects=[{"class_name": "bottle"}])
    scene = {"description": "A bottle is visible.", "visible_object_count": 1, "warnings": []}
    first_query = prepare_spoken_response(query)
    first_scene = prepare_spoken_scene_summary(scene)

    assert set(first_query) == {"spoken_text", "source_status", "warnings"}
    assert set(first_scene) == {"spoken_text", "source_status", "warnings"}
    assert first_query == prepare_spoken_response(query)
    assert first_scene == prepare_spoken_scene_summary(scene)