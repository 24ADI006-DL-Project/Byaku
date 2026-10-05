from byaku_rebuild.scene_graph import (
    build_scene_graph,
    find_objects_by_class,
    find_relationships_for_object,
    get_objects,
    query_relationship,
)


def make_object(
    object_id: str,
    class_name: str,
    *,
    track_id: int | None = None,
    relative_depth_value: float | None = None,
) -> dict:
    item = {
        "object_id": object_id,
        "track_id": track_id,
        "class_name": class_name,
        "bounding_box": (1.0, 2.0, 11.0, 12.0),
        "center": (6.0, 7.0),
        "horizontal_position": "left",
        "vertical_position": "above",
    }
    if relative_depth_value is not None:
        item["relative_depth_value"] = relative_depth_value
    return item


def test_builds_directed_graph_and_preserves_node_and_edge_evidence() -> None:
    spatial_output = {
        "image_shape": (100, 120),
        "objects": [
            make_object("track:3", "bottle", track_id=3, relative_depth_value=0.82),
            make_object("track:8", "chair", track_id=8, relative_depth_value=0.24),
        ],
        "relationships": [
            {"subject": "track:3", "relation": "left_of", "object": "track:8"},
            {
                "subject": "track:3",
                "relation": "nearer_than",
                "object": "track:8",
                "depth_difference": 0.58,
            },
            {"subject": "track:3", "relation": "overlapping", "object": "track:8", "iou": 0.2},
        ],
        "warnings": [],
    }

    graph = build_scene_graph(spatial_output)

    assert graph.is_directed()
    assert graph.number_of_nodes() == 2
    assert graph.number_of_edges() == 3
    assert graph.nodes["track:3"]["class_name"] == "bottle"
    assert graph.nodes["track:3"]["bounding_box"] == (1.0, 2.0, 11.0, 12.0)
    assert graph.nodes["track:3"]["center"] == (6.0, 7.0)
    assert graph.nodes["track:3"]["relative_depth_value"] == 0.82
    assert graph.edges["track:3", "track:8", "nearer_than"]["depth_difference"] == 0.58
    assert graph.edges["track:3", "track:8", "overlapping"]["iou"] == 0.2


def test_directional_queries_preserve_source_and_destination() -> None:
    graph = build_scene_graph(
        {
            "objects": [make_object("a", "person"), make_object("b", "chair")],
            "relationships": [
                {"subject": "a", "relation": "left_of", "object": "b"},
                {"subject": "b", "relation": "right_of", "object": "a"},
            ],
        }
    )

    assert query_relationship(graph, "a", "b", "left_of")[0]["relation"] == "left_of"
    assert query_relationship(graph, "b", "a", "right_of")[0]["relation"] == "right_of"
    assert query_relationship(graph, "b", "a", "left_of") == []
    assert {item["direction"] for item in find_relationships_for_object(graph, "a")} == {"outgoing", "incoming"}


def test_get_objects_and_find_by_class() -> None:
    graph = build_scene_graph(
        {
            "objects": [make_object("one", "Bottle"), make_object("two", "bottle"), make_object("three", "cup")],
            "relationships": [],
        }
    )

    assert len(get_objects(graph)) == 3
    assert {item["object_id"] for item in find_objects_by_class(graph, "BOTTLE")} == {"one", "two"}
    assert find_objects_by_class(graph, "person") == []


def test_objects_without_relationships_remain_in_graph() -> None:
    graph = build_scene_graph({"objects": [make_object("solo", "person")], "relationships": []})

    assert graph.number_of_nodes() == 1
    assert graph.number_of_edges() == 0
    assert get_objects(graph)[0]["object_id"] == "solo"
    assert find_relationships_for_object(graph, "solo") == []


def test_empty_or_invalid_spatial_input_produces_safe_empty_graph() -> None:
    for spatial_output in ({"objects": [], "relationships": []}, None, []):
        graph = build_scene_graph(spatial_output)
        assert graph.number_of_nodes() == 0
        assert graph.number_of_edges() == 0
        assert get_objects(graph) == []

    assert query_relationship(None, "a", "b") == []
    assert find_objects_by_class(None, "bottle") == []
    assert find_relationships_for_object(None, "a") == []


def test_duplicate_ids_keep_first_node_and_duplicate_edges_are_skipped() -> None:
    graph = build_scene_graph(
        {
            "objects": [
                make_object("same", "bottle", track_id=1),
                make_object("same", "cup", track_id=2),
                make_object("other", "chair"),
            ],
            "relationships": [
                {"subject": "same", "relation": "left_of", "object": "other"},
                {"subject": "same", "relation": "left_of", "object": "other"},
                {"subject": "same", "relation": "above", "object": "other"},
            ],
        }
    )

    assert graph.number_of_nodes() == 2
    assert graph.nodes["same"]["class_name"] == "bottle"
    assert graph.number_of_edges() == 2
    assert any("duplicate object ID" in message for message in graph.graph["warnings"])
    assert any("duplicate 'left_of'" in message for message in graph.graph["warnings"])


def test_invalid_relationship_references_and_types_are_skipped() -> None:
    graph = build_scene_graph(
        {
            "objects": [make_object("a", "person"), make_object("b", "chair")],
            "relationships": [
                {"subject": "a", "relation": "left_of", "object": "missing"},
                {"subject": "a", "relation": "unsupported", "object": "b"},
                {"subject": "a", "relation": "above", "object": "a"},
                {"subject": "a", "object": "b"},
                "not a relationship mapping",
            ],
        }
    )

    assert graph.number_of_nodes() == 2
    assert graph.number_of_edges() == 0
    assert len(graph.graph["warnings"]) == 5


def test_relationship_types_between_same_pair_are_distinct_edges() -> None:
    graph = build_scene_graph(
        {
            "objects": [make_object("a", "bottle"), make_object("b", "table")],
            "relationships": [
                {"subject": "a", "relation": "left_of", "object": "b"},
                {"subject": "a", "relation": "above", "object": "b"},
                {"subject": "a", "relation": "similar_depth", "object": "b", "depth_difference": 0.01},
            ],
        }
    )

    assert graph.number_of_edges("a", "b") == 3
    assert {item["relation"] for item in query_relationship(graph, "a", "b")} == {
        "left_of",
        "above",
        "similar_depth",
    }


def test_missing_optional_depth_is_preserved_as_missing() -> None:
    graph = build_scene_graph(
        {
            "frame_id": 3,
            "objects": [make_object("frame:0", "person")],
            "relationships": [],
        }
    )

    assert graph.nodes["temporary:3:0"]["class_name"] == "person"
    assert "relative_depth_value" not in graph.nodes["temporary:3:0"]


def test_track_id_can_supply_node_id_when_object_id_is_missing() -> None:
    graph = build_scene_graph(
        {
            "objects": [
                {"track_id": 6, "class_name": "person"},
                {"class_name": "chair"},
            ],
            "frame_id": 12,
            "relationships": [],
        }
    )

    assert graph.nodes["track:6"]["track_id"] == 6
    assert graph.nodes["temporary:12:1"]["class_name"] == "chair"


def test_legacy_frame_ids_and_relationship_references_are_normalized() -> None:
    graph = build_scene_graph(
        {
            "frame_id": 22,
            "objects": [make_object("frame:0", "bottle"), make_object("frame:1", "chair")],
            "relationships": [{"subject": "frame:0", "relation": "left_of", "object": "frame:1"}],
        }
    )

    assert set(graph.nodes) == {"temporary:22:0", "temporary:22:1"}
    assert graph.has_edge("temporary:22:0", "temporary:22:1", key="left_of")