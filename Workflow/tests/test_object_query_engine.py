from Workflow.reasoning.object_query_engine import answer_query
from Workflow.reasoning.scene_relationship_graph import build_scene_graph
from Workflow.reasoning.scene_memory import SceneMemory


def make_object(
    object_id: str,
    class_name: str,
    box: tuple[float, float, float, float],
    *,
    depth: float | None = None,
    horizontal: str | None = None,
    vertical: str | None = None,
    status: str | None = None,
) -> dict:
    item = {
        "object_id": object_id,
        "class_name": class_name,
        "bounding_box": box,
        "center": ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2),
    }
    if depth is not None:
        item["relative_depth_value"] = depth
    if horizontal is not None:
        item["horizontal_position"] = horizontal
    if vertical is not None:
        item["vertical_position"] = vertical
    if status is not None:
        item["status"] = status
    return item


def test_object_location_query_uses_supplied_position_and_relationships() -> None:
    spatial = {
        "objects": [
            make_object("bottle-1", "bottle", (0, 0, 10, 10), depth=0.8, horizontal="left"),
            make_object("chair-1", "chair", (20, 0, 30, 10), depth=0.2),
        ],
        "relationships": [
            {"subject": "bottle-1", "relation": "left_of", "object": "chair-1"},
            {"subject": "bottle-1", "relation": "nearer_than", "object": "chair-1", "depth_difference": 0.6},
        ],
    }

    result = answer_query("Where is my bottle?", spatial_output=spatial)

    assert result["status"] == "visible"
    assert result["matched_objects"][0]["class_name"] == "bottle"
    assert "to the left of a chair" in result["answer"]
    assert "in front of a chair" in result["answer"]


def test_where_is_person_grounded_in_camera_perspective() -> None:
    spatial = {
        "objects": [
            make_object("person-4", "person", (200, 100, 440, 380), horizontal="center", vertical="middle"),
        ],
        "relationships": [],
    }

    result = answer_query("Where is the person?", spatial_output=spatial)

    assert result["status"] == "visible"
    assert result["matched_objects"][0]["class_name"] == "person"
    assert "in front of you, near the center" in result["answer"]


def test_visible_object_listing_uses_scene_description_engine() -> None:
    spatial = {
        "objects": [make_object("a", "bottle", (0, 0, 10, 10)), make_object("b", "chair", (20, 0, 30, 10))],
        "relationships": [],
    }

    result = answer_query("What objects are visible?", spatial_output=spatial)

    assert result["status"] == "visible"
    assert "A bottle is visible." in result["answer"]
    assert "A chair is visible." in result["answer"]
    assert len(result["matched_objects"]) == 2


def test_left_and_right_relationship_questions_use_directional_edges() -> None:
    spatial = {
        "objects": [make_object("b", "bottle", (0, 0, 10, 10)), make_object("c", "chair", (20, 0, 30, 10))],
        "relationships": [
            {"subject": "b", "relation": "left_of", "object": "c"},
            {"subject": "c", "relation": "right_of", "object": "b"},
        ],
    }

    left = answer_query("What is to the left of the chair?", spatial_output=spatial)
    right = answer_query("What is to the right of the bottle?", spatial_output=spatial)

    assert left["status"] == "visible"
    assert left["matched_objects"][0]["class_name"] == "bottle"
    assert right["status"] == "visible"
    assert right["matched_objects"][0]["class_name"] == "chair"


def test_front_and_behind_queries_require_relative_depth_evidence() -> None:
    spatial = {
        "objects": [make_object("p", "person", (0, 0, 10, 10), depth=0.9), make_object("t", "table", (20, 0, 30, 10), depth=0.2)],
        "relationships": [
            {"subject": "p", "relation": "nearer_than", "object": "t", "depth_difference": 0.7},
            {"subject": "t", "relation": "farther_than", "object": "p", "depth_difference": 0.7},
        ],
    }

    in_front = answer_query("What is in front of the table?", spatial_output=spatial)
    behind = answer_query("What is behind the person?", spatial_output=spatial)

    assert in_front["matched_objects"][0]["class_name"] == "person"
    assert "in front of a table" in in_front["answer"]
    assert behind["matched_objects"][0]["class_name"] == "table"
    assert "behind a person" in behind["answer"]


def test_current_visibility_and_temporarily_missing_state_are_distinct() -> None:
    memory = SceneMemory(stale_after_frames=4)
    observation = {"track_id": 3, "class_name": "bottle", "bounding_box": (1, 2, 11, 12)}
    memory.update([observation], frame_number=1, timestamp=10.0)
    visible = answer_query("Is the bottle currently visible?", scene_memory=memory)
    memory.update([], frame_number=2, timestamp=11.0)
    missing = answer_query("Is the bottle currently visible?", scene_memory=memory)

    assert visible["status"] == "visible"
    assert "currently visible" in visible["answer"]
    assert missing["status"] == "missing"
    assert "temporarily missing" in missing["answer"]
    assert "frame 1" in missing["answer"]
    assert missing["matched_objects"][0]["status"] == "missing"


def test_stale_object_is_not_claimed_visible() -> None:
    memory = SceneMemory(stale_after_frames=2)
    memory.update([{"track_id": 8, "class_name": "chair", "bounding_box": (5, 5, 15, 15)}], frame_number=1)
    memory.update([], frame_number=2)
    memory.update([], frame_number=3)

    result = answer_query("Where is the chair?", scene_memory=memory)

    assert result["status"] == "stale"
    assert "stale" in result["answer"]
    assert "currently visible" not in result["answer"]
    assert result["matched_objects"][0]["status"] == "stale"


def test_last_seen_query_uses_scene_memory_frame_and_box() -> None:
    memory = SceneMemory()
    memory.update(
        [{"track_id": 2, "class_name": "chair", "bounding_box": (10, 20, 30, 40)}],
        frame_number=12,
        timestamp=123.5,
    )
    memory.update([], frame_number=13, timestamp=124.0)

    result = answer_query("Where was the chair last seen?", scene_memory=memory)

    assert result["status"] == "missing"
    assert "frame 12" in result["answer"]
    assert "(10.0, 20.0, 30.0, 40.0) pixels" in result["answer"]
    assert result["evidence"]["object_state"]["last_seen_timestamp"] == 123.5


def test_unknown_object_is_not_found() -> None:
    result = answer_query("Where is my umbrella?", spatial_output={"objects": [], "relationships": []})

    assert result["status"] == "not_found"
    assert "can't find a umbrella" in result["answer"]
    assert result["matched_objects"] == []


def test_multiple_matching_instances_are_reported_as_ambiguous() -> None:
    spatial = {
        "objects": [make_object("b1", "bottle", (0, 0, 10, 10)), make_object("b2", "bottle", (20, 0, 30, 10))],
        "relationships": [],
    }

    result = answer_query("Where is the bottle?", spatial_output=spatial)

    assert result["status"] == "ambiguous"
    assert "multiple bottle instances" in result["answer"]
    assert len(result["matched_objects"]) == 2


def test_empty_scene_and_missing_relationship_data_are_clear() -> None:
    empty = answer_query("What objects are visible?", spatial_output={"objects": [], "relationships": []})
    missing_relationships = answer_query(
        "What is to the left of the chair?",
        spatial_output={"objects": [make_object("c", "chair", (0, 0, 10, 10))]},
    )

    assert empty["status"] == "empty_scene"
    assert empty["answer"] == "No objects are currently visible."
    assert missing_relationships["status"] == "insufficient_evidence"
    assert "relationship data" in missing_relationships["answer"]


def test_scene_graph_relationships_are_reused() -> None:
    graph = build_scene_graph(
        {
            "objects": [make_object("p", "person", (0, 0, 10, 10)), make_object("t", "table", (20, 0, 30, 10))],
            "relationships": [{"subject": "p", "relation": "left_of", "object": "t"}],
        }
    )

    result = answer_query("What is to the left of a table?", scene_graph=graph)

    assert result["matched_objects"][0]["class_name"] == "person"
    assert result["evidence"]["relationships"][0]["relation"] == "left_of"


def test_malformed_input_and_invalid_relationship_references_are_safe() -> None:
    malformed = answer_query("Where is the bottle?", spatial_output={"objects": [None], "relationships": []})
    invalid_reference = answer_query(
        "What is to the left of the chair?",
        spatial_output={
            "objects": [make_object("c", "chair", (20, 0, 30, 10))],
            "relationships": [{"subject": "unknown", "relation": "left_of", "object": "c"}],
        },
    )

    assert malformed["status"] == "not_found"
    assert malformed["warnings"]
    assert invalid_reference["status"] == "not_found"
    assert any("non-visible" in warning for warning in invalid_reference["warnings"])


def test_unsupported_query_and_missing_location_evidence_do_not_invent_claims() -> None:
    unsupported = answer_query("What color is the bottle?", spatial_output={"objects": [make_object("b", "bottle", (0, 0, 10, 10))]})
    unknown_location = answer_query("Where is the bottle?", spatial_output={"objects": [{"object_id": "b", "class_name": "bottle"}]})

    assert unsupported["status"] == "unsupported"
    assert unknown_location["status"] == "insufficient_evidence"
    assert "no supported location" in unknown_location["answer"]
    assert "meter" not in unsupported["answer"].lower()
    assert "meter" not in unknown_location["answer"].lower()


def test_duplicate_and_malformed_relationships_are_not_reported_as_evidence() -> None:
    spatial = {
        "objects": [make_object("b", "bottle", (0, 0, 10, 10)), make_object("c", "chair", (20, 0, 30, 10))],
        "relationships": [
            {"subject": "b", "relation": "left_of", "object": "c"},
            {"subject": "b", "relation": "left_of", "object": "c"},
            None,
        ],
    }

    result = answer_query("What is to the left of the chair?", spatial_output=spatial)

    assert len(result["evidence"]["relationships"]) == 1
    assert any("duplicate" in warning for warning in result["warnings"])
    assert any("malformed" in warning for warning in result["warnings"])


def test_output_schema_and_identical_inputs_are_deterministic() -> None:
    spatial = {
        "objects": [make_object("b", "bottle", (0, 0, 10, 10), horizontal="left")],
        "relationships": [],
    }
    first = answer_query("Where is bottle?", spatial_output=spatial)
    second = answer_query("Where is bottle?", spatial_output=spatial)

    assert set(first) == {"query", "answer", "status", "matched_objects", "evidence", "warnings"}
    assert first == second