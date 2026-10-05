from byaku_rebuild.scene_description import describe_scene
from byaku_rebuild.scene_graph import build_scene_graph
from byaku_rebuild.scene_memory import SceneMemory


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


def test_describes_multiple_objects_and_supported_relationships() -> None:
    spatial = {
        "objects": [
            make_object("track:1", "chair", (0, 0, 20, 20), depth=0.8),
            make_object("track:2", "table", (15, 10, 35, 30), depth=0.2),
        ],
        "relationships": [
            {"subject": "track:1", "relation": "left_of", "object": "track:2"},
            {"subject": "track:2", "relation": "right_of", "object": "track:1"},
            {"subject": "track:1", "relation": "above", "object": "track:2"},
            {"subject": "track:2", "relation": "below", "object": "track:1"},
            {
                "subject": "track:1",
                "relation": "nearer_than",
                "object": "track:2",
                "depth_difference": 0.6,
            },
            {"subject": "track:1", "relation": "overlapping", "object": "track:2", "iou": 0.1},
        ],
    }

    result = describe_scene(spatial)

    assert result["visible_object_count"] == 2
    assert "A chair is to the left of a table." in result["description"]
    assert "A chair is above a table." in result["description"]
    assert "A chair is in front of a table." in result["description"]
    assert "A chair overlaps a table." in result["description"]
    assert len(result["described_relationships"]) == 4
    assert result["warnings"] == []


def test_one_object_uses_class_and_supplied_position() -> None:
    result = describe_scene(
        {
            "objects": [make_object("track:5", "bottle", (80, 80, 100, 100), horizontal="right", vertical="below")],
            "relationships": [],
        }
    )

    assert result["visible_object_count"] == 1
    assert result["description"] == "A bottle is visible in the lower right."
    assert result["described_relationships"] == []


def test_empty_scene_has_stable_empty_description_and_structured_fields() -> None:
    result = describe_scene({"objects": [], "relationships": []})

    assert set(result) == {"description", "visible_object_count", "described_relationships", "warnings"}
    assert result["description"] == "No objects are currently visible."
    assert result["visible_object_count"] == 0
    assert result["described_relationships"] == []


def test_scene_memory_excludes_missing_and_stale_objects() -> None:
    memory = SceneMemory(stale_after_frames=2)
    memory.update(
        [
            {"track_id": 1, "class_name": "chair", "bounding_box": (0, 0, 10, 10)},
            {"track_id": 2, "class_name": "bottle", "bounding_box": (10, 0, 20, 10)},
            {"track_id": 3, "class_name": "table", "bounding_box": (20, 0, 30, 10)},
        ],
        frame_number=1,
    )
    memory.update(
        [
            {"track_id": 1, "class_name": "chair", "bounding_box": (0, 0, 10, 10)},
            {"track_id": 2, "class_name": "bottle", "bounding_box": (10, 0, 20, 10)},
        ],
        frame_number=2,
    )
    memory.update(
        [{"track_id": 1, "class_name": "chair", "bounding_box": (0, 0, 10, 10)}],
        frame_number=3,
    )
    result = describe_scene(scene_memory=memory)

    assert result["visible_object_count"] == 1
    assert result["description"] == "A chair is visible."
    assert "bottle" not in result["description"]
    assert "table" not in result["description"]
    assert any("1 temporarily missing and 1 stale" in warning for warning in result["warnings"])


def test_memory_snapshot_status_is_authoritative_for_relationship_visibility() -> None:
    spatial = {
        "objects": [make_object("track:1", "chair", (0, 0, 10, 10)), make_object("track:2", "table", (20, 0, 30, 10))],
        "relationships": [{"subject": "track:1", "relation": "left_of", "object": "track:2"}],
    }
    memory_snapshot = {
        "visible_objects": [
            {"object_id": "track:1", "class_name": "chair", "status": "visible", "bounding_box": (0, 0, 10, 10)}
        ],
        "temporarily_missing_objects": [
            {"object_id": "track:2", "class_name": "table", "status": "missing"}
        ],
        "stale_objects": [],
    }

    result = describe_scene(spatial, scene_memory=memory_snapshot)

    assert result["description"] == "A chair is visible."
    assert result["described_relationships"] == []
    assert any("non-visible" in warning for warning in result["warnings"])


def test_missing_depth_prevents_front_behind_language() -> None:
    result = describe_scene(
        {
            "objects": [
                make_object("a", "person", (0, 0, 10, 10)),
                make_object("b", "chair", (20, 0, 30, 10)),
            ],
            "relationships": [{"subject": "a", "relation": "nearer_than", "object": "b", "depth_difference": 0.5}],
        }
    )

    assert result["described_relationships"] == []
    assert "front" not in result["description"]
    assert "behind" not in result["description"]
    assert any("relative-depth evidence" in warning for warning in result["warnings"])


def test_missing_relationships_fall_back_to_object_list() -> None:
    result = describe_scene(
        {"objects": [make_object("a", "person", (0, 0, 10, 10)), make_object("b", "chair", (20, 0, 30, 10))]}
    )

    assert result["visible_object_count"] == 2
    assert "A person is visible." in result["description"]
    assert "A chair is visible." in result["description"]
    assert result["described_relationships"] == []


def test_duplicate_reciprocal_relationships_are_described_once() -> None:
    spatial = {
        "objects": [make_object("a", "person", (0, 0, 10, 10)), make_object("b", "chair", (20, 0, 30, 10))],
        "relationships": [
            {"subject": "a", "relation": "left_of", "object": "b"},
            {"subject": "b", "relation": "right_of", "object": "a"},
            {"subject": "a", "relation": "left_of", "object": "b"},
        ],
    }

    result = describe_scene(spatial)

    assert result["description"] == "A person is to the left of a chair."
    assert len(result["described_relationships"]) == 1
    assert len([warning for warning in result["warnings"] if "duplicate" in warning]) == 1


def test_malformed_objects_and_invalid_relationship_references_are_ignored() -> None:
    result = describe_scene(
        {
            "objects": [
                make_object("a", "person", (0, 0, 10, 10)),
                {"object_id": "bad", "class_name": "", "bounding_box": (0, 0, 2, 2)},
                make_object("a", "chair", (20, 0, 30, 10)),
                None,
            ],
            "relationships": [
                {"subject": "a", "relation": "left_of", "object": "missing"},
                {"subject": "a", "relation": "holding", "object": "a"},
                None,
            ],
        }
    )

    assert result["visible_object_count"] == 1
    assert result["description"] == "A person is visible."
    assert result["described_relationships"] == []
    assert len(result["warnings"]) >= 5


def test_contradictory_or_unsupported_relationships_are_not_described() -> None:
    result = describe_scene(
        {
            "objects": [
                make_object("a", "person", (25, 0, 35, 10)),
                make_object("b", "chair", (5, 0, 15, 10)),
            ],
            "relationships": [
                {"subject": "a", "relation": "left_of", "object": "b"},
                {"subject": "a", "relation": "holding", "object": "b"},
            ],
        }
    )

    assert result["described_relationships"] == []
    assert "left of" not in result["description"]
    assert "holding" not in result["description"]
    assert "meter" not in result["description"].lower()
    assert result["visible_object_count"] == 2


def test_scene_graph_input_uses_supplied_edges_and_nodes() -> None:
    graph = build_scene_graph(
        {
            "objects": [
                make_object("track:1", "bottle", (0, 0, 10, 10), depth=0.8),
                make_object("track:2", "chair", (20, 0, 30, 10), depth=0.2),
            ],
            "relationships": [
                {"subject": "track:1", "relation": "nearer_than", "object": "track:2", "depth_difference": 0.6}
            ],
        }
    )

    result = describe_scene(scene_graph=graph)

    assert result["visible_object_count"] == 2
    assert result["description"] == "A bottle is in front of a chair."
    assert result["described_relationships"][0]["depth_difference"] == 0.6


def test_description_is_deterministic_for_identical_input() -> None:
    spatial = {
        "objects": [make_object("a", "bottle", (0, 0, 10, 10), horizontal="left")],
        "relationships": [],
    }

    assert describe_scene(spatial) == describe_scene(spatial)


def test_duplicate_object_ids_and_stale_status_do_not_claim_visibility() -> None:
    result = describe_scene(
        {
            "objects": [
                make_object("track:1", "chair", (0, 0, 10, 10)),
                make_object("track:1", "table", (20, 0, 30, 10)),
                make_object("track:2", "bottle", (30, 0, 40, 10), status="stale"),
            ],
            "relationships": [],
        }
    )

    assert result["visible_object_count"] == 1
    assert result["description"] == "A chair is visible."
    assert any("duplicate visible object ID" in warning for warning in result["warnings"])
    assert any("status 'stale'" in warning for warning in result["warnings"])