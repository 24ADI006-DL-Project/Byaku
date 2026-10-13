import numpy as np

from Workflow.reasoning.object_query_engine import answer_query
from Workflow.reasoning.scene_description import describe_scene
from Workflow.reasoning.scene_relationship_graph import build_scene_graph
from Workflow.reasoning.scene_memory import SceneMemory
from Workflow.reasoning.spatial_relation_engine import calculate_spatial_relationships
from Workflow.reasoning.narration_response import (
    prepare_spoken_response,
    prepare_spoken_scene_summary,
)


def make_tracked_detections() -> list[dict]:
    return [
        {
            "track_id": 7,
            "class_name": "bottle",
            "confidence": 0.9,
            "bounding_box": (2, 2, 12, 12),
        },
        {
            "track_id": 9,
            "class_name": "chair",
            "confidence": 0.8,
            "bounding_box": (25, 2, 35, 12),
        },
    ]


def test_tracked_frame_contract_flows_through_query_and_speech() -> None:
    depth_map = np.zeros((20, 40), dtype=np.float32)
    depth_map[:, :20] = 0.8
    depth_map[:, 20:] = 0.2
    spatial_output = calculate_spatial_relationships(
        make_tracked_detections(),
        (20, 40),
        depth_map,
    )
    graph = build_scene_graph(spatial_output)
    memory = SceneMemory()
    memory_snapshot = memory.update(spatial_output, frame_number=1, timestamp=1.0)

    spatial_ids = {item["object_id"] for item in spatial_output["objects"]}
    graph_ids = set(graph.nodes)
    memory_ids = {item["object_id"] for item in memory_snapshot["visible_objects"]}
    assert spatial_ids == graph_ids == memory_ids == {"track:7", "track:9"}

    scene_summary = describe_scene(spatial_output, scene_graph=graph, scene_memory=memory)
    query_result = answer_query(
        "What is to the left of the chair?",
        spatial_output=spatial_output,
        scene_graph=graph,
        scene_memory=memory,
    )
    spoken_query = prepare_spoken_response(query_result)
    spoken_scene = prepare_spoken_scene_summary(scene_summary)

    assert scene_summary["visible_object_count"] == 2
    assert query_result["status"] == "visible"
    assert "bottle" in query_result["answer"] and "chair" in query_result["answer"]
    assert spoken_query["source_status"] == "visible"
    assert "to the left of a chair" in spoken_query["spoken_text"]
    assert spoken_scene["spoken_text"] == scene_summary["description"]


def test_valid_empty_frame_propagates_without_claiming_visibility() -> None:
    spatial_output = calculate_spatial_relationships([], (20, 40))
    graph = build_scene_graph(spatial_output)
    memory = SceneMemory()
    memory.update(spatial_output, frame_number=1)

    scene_summary = describe_scene(spatial_output, scene_graph=graph, scene_memory=memory)
    query_result = answer_query(
        "What objects are visible?",
        spatial_output=spatial_output,
        scene_graph=graph,
        scene_memory=memory,
    )

    assert spatial_output["objects"] == []
    assert graph.number_of_nodes() == 0
    assert scene_summary["visible_object_count"] == 0
    assert query_result["status"] == "empty_scene"
    assert prepare_spoken_scene_summary(scene_summary)["spoken_text"] == "No objects are currently visible."
    assert prepare_spoken_response(query_result)["spoken_text"] == "No objects are currently visible."


def test_memory_visibility_is_authoritative_after_object_disappears() -> None:
    spatial_output = calculate_spatial_relationships(make_tracked_detections(), (20, 40))
    graph = build_scene_graph(spatial_output)
    memory = SceneMemory(stale_after_frames=3)
    memory.update(spatial_output, frame_number=1)
    missing_snapshot = memory.update([], frame_number=2)

    query_result = answer_query(
        "Is the bottle currently visible?",
        spatial_output=spatial_output,
        scene_graph=graph,
        scene_memory=memory,
    )
    spoken = prepare_spoken_response(query_result)

    assert len(missing_snapshot["temporarily_missing_objects"]) == 2
    assert query_result["status"] == "missing"
    assert "not currently visible" in spoken["spoken_text"]
    assert "is currently visible" not in spoken["spoken_text"]


def test_untracked_ids_match_across_spatial_graph_and_memory_per_frame() -> None:
    detections = [
        {"class_name": "bottle", "bounding_box": (2, 2, 12, 12)},
        {"class_name": "chair", "bounding_box": (25, 2, 35, 12)},
    ]
    spatial_output = calculate_spatial_relationships(detections, (20, 40), frame_id=17)
    graph = build_scene_graph(spatial_output)
    memory = SceneMemory()
    memory_snapshot = memory.update(spatial_output, frame_number=17)

    spatial_ids = {item["object_id"] for item in spatial_output["objects"]}
    graph_ids = set(graph.nodes)
    memory_ids = {item["object_id"] for item in memory_snapshot["visible_objects"]}
    assert spatial_ids == graph_ids == memory_ids == {"temporary:17:0", "temporary:17:1"}

    scene_summary = describe_scene(spatial_output, scene_graph=graph, scene_memory=memory)
    assert scene_summary["visible_object_count"] == 2
    assert any(item["relation"] == "left_of" for item in scene_summary["described_relationships"])

    next_spatial_output = calculate_spatial_relationships(detections, (20, 40), frame_id=18)
    next_snapshot = memory.update(next_spatial_output, frame_number=18)
    next_ids = {item["object_id"] for item in next_snapshot["visible_objects"]}
    assert next_ids == {"temporary:18:0", "temporary:18:1"}
    assert next_ids.isdisjoint(spatial_ids)


def test_confidence_survives_spatial_memory_and_query_adapters() -> None:
    detections = [
        {"track_id": 7, "class_name": "bottle", "confidence": 0.91, "bounding_box": (2, 2, 12, 12)},
        {"track_id": 9, "class_name": "chair", "bounding_box": (25, 2, 35, 12)},
    ]
    spatial_output = calculate_spatial_relationships(detections, (20, 40), frame_id=31)
    memory = SceneMemory()
    memory.update(spatial_output, frame_number=31)
    result = answer_query("Where is the bottle?", spatial_output=spatial_output, scene_memory=memory)

    bottle = next(item for item in spatial_output["objects"] if item["class_name"] == "bottle")
    assert bottle["confidence"] == 0.91
    assert memory.get_object(7)["confidence"] == 0.91
    assert result["matched_objects"][0]["confidence"] == 0.91
    assert "confidence" not in next(item for item in spatial_output["objects"] if item["class_name"] == "chair")


def test_evicted_stale_object_is_not_returned_by_last_seen_query() -> None:
    memory = SceneMemory(stale_after_frames=2, stale_retention_frames=3)
    memory.update(
        [{"track_id": 12, "class_name": "bottle", "bounding_box": (2, 2, 12, 12)}],
        frame_number=1,
        timestamp=1.0,
    )
    memory.update([], frame_number=2)
    stale = memory.update([], frame_number=3)
    still_retained = memory.update([], frame_number=4)

    assert [item["object_id"] for item in stale["stale_objects"]] == ["track:12"]
    assert [item["object_id"] for item in still_retained["stale_objects"]] == ["track:12"]
    assert memory.update([], frame_number=5)["stale_objects"] == []
    result = answer_query("Where was the bottle last seen?", scene_memory=memory)

    assert result["status"] == "not_found"
    assert result["matched_objects"] == []
    assert "can't find" in result["answer"]


def test_mismatched_depth_is_skipped_but_2d_relationships_remain() -> None:
    depth_map = np.ones((10, 10), dtype=np.float32)
    spatial_output = calculate_spatial_relationships(make_tracked_detections(), (20, 40), depth_map)

    assert all(item["relative_depth_value"] is None for item in spatial_output["objects"])
    assert any(item["relation"] == "left_of" for item in spatial_output["relationships"])
    assert not any(item["relation"] in {"nearer_than", "farther_than", "similar_depth"}
                   for item in spatial_output["relationships"])
    assert any("do not match" in warning for warning in spatial_output["warnings"])