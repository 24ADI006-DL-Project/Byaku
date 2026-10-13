import numpy as np
import Workflow.scene_pipeline as pipeline_module

from Workflow.scene_pipeline import PipelineOrchestrator
from Workflow.reasoning.scene_memory import SceneMemory


def detection(
    class_name: str,
    box: tuple[float, float, float, float],
    *,
    track_id: int | None = None,
    confidence: float | None = None,
) -> dict:
    result = {"class_name": class_name, "bounding_box": box}
    if track_id is not None:
        result["track_id"] = track_id
    if confidence is not None:
        result["confidence"] = confidence
    return result


def process(
    pipeline: PipelineOrchestrator,
    frame_id: str,
    frame_number: int,
    detections: list[dict],
    depth_map: np.ndarray | None = None,
) -> dict:
    return pipeline.process_frame(
        frame_id=frame_id,
        frame_number=frame_number,
        image_dimensions={"width": 40, "height": 20},
        detections=detections,
        depth_map=depth_map,
        timestamp=float(frame_number),
    )


def test_multiple_detections_generate_relationships_graph_description_and_speech() -> None:
    pipeline = PipelineOrchestrator()
    depth = np.zeros((20, 40), dtype=np.float32)
    depth[:, :20] = 0.8
    depth[:, 20:] = 0.2

    result = process(
        pipeline,
        "frame-A",
        1,
        [
            detection("bottle", (2, 2, 12, 12), track_id=7, confidence=0.91),
            detection("chair", (25, 2, 35, 12), track_id=9, confidence=0.82),
        ],
        depth,
    )

    assert result["status"] == "success"
    assert result["frame"] == {
        "frame_id": "frame-A",
        "frame_number": 1,
        "timestamp": 1.0,
        "image_shape": (20, 40),
        "width": 40,
        "height": 20,
    }
    assert [item["object_id"] for item in result["objects"]] == ["track:7", "track:9"]
    assert result["objects"][0]["confidence"] == 0.91
    assert any(item["relation"] == "left_of" for item in result["relationships"])
    assert any(item["relation"] == "nearer_than" for item in result["relationships"])
    assert {item["object_id"] for item in result["graph"]["nodes"]} == {"track:7", "track:9"}
    assert result["scene_description"]["visible_object_count"] == 2
    assert result["scene_summary_speech"]["source_status"] == "visible"

    query_result = pipeline.query("What is to the left of the chair?")
    assert query_result["status"] == "visible"
    assert "bottle" in query_result["query_result"]["answer"]
    assert "to the left of a chair" in query_result["speech_response"]["spoken_text"]


def test_tracked_ids_and_observation_history_remain_stable_across_frames() -> None:
    pipeline = PipelineOrchestrator()
    first = process(pipeline, "f1", 1, [detection("bottle", (2, 2, 12, 12), track_id=7, confidence=0.8)])
    second = process(pipeline, "f2", 2, [detection("bottle", (4, 2, 14, 12), track_id=7, confidence=0.9)])

    assert first["graph"]["nodes"][0]["object_id"] == "track:7"
    assert second["graph"]["nodes"][0]["object_id"] == "track:7"
    assert second["memory"]["visible_objects"][0]["last_seen_frame"] == 2
    assert [item["frame_number"] for item in pipeline.scene_memory.get_recent_observations(7)] == [1, 2]
    assert second["memory"]["visible_objects"][0]["confidence"] == 0.9


def test_untracked_ids_are_canonical_unique_and_resolve_in_graph_and_memory() -> None:
    pipeline = PipelineOrchestrator()
    first = process(pipeline, "untracked-1", 1, [detection("bottle", (2, 2, 12, 12))])
    second = process(pipeline, "untracked-2", 2, [detection("bottle", (3, 2, 13, 12))])

    first_id = first["objects"][0]["object_id"]
    second_id = second["objects"][0]["object_id"]
    assert first_id == "temporary:untracked-1:0"
    assert second_id == "temporary:untracked-2:0"
    assert first_id != second_id
    assert first_id == first["graph"]["nodes"][0]["object_id"]
    assert first_id == first["memory"]["visible_objects"][0]["object_id"]


def test_queries_distinguish_visible_missing_stale_and_evicted_unknown_objects() -> None:
    memory = SceneMemory(stale_after_frames=2, stale_retention_frames=3)
    pipeline = PipelineOrchestrator(memory)
    process(pipeline, "visible", 1, [detection("bottle", (2, 2, 12, 12), track_id=7)])
    visible = pipeline.query("Is the bottle currently visible?")
    process(pipeline, "missing", 2, [])
    missing = pipeline.query("Is the bottle currently visible?")
    process(pipeline, "stale", 3, [])
    stale = pipeline.query("Where was the bottle last seen?")
    process(pipeline, "retained", 4, [])
    retained = pipeline.query("Where is the bottle?")
    process(pipeline, "evicted", 5, [])
    evicted = pipeline.query("Where was the bottle last seen?")

    assert visible["status"] == "visible"
    assert missing["status"] == "missing"
    assert "not currently visible" in missing["speech_response"]["spoken_text"]
    assert stale["status"] == "stale"
    assert "last-known location may no longer be reliable" in stale["speech_response"]["spoken_text"]
    assert retained["status"] == "stale"
    assert evicted["status"] == "not_found"


def test_empty_frame_missing_depth_and_latest_scene_summary() -> None:
    pipeline = PipelineOrchestrator()
    result = process(pipeline, "empty", 1, [], depth_map=None)
    scene = pipeline.describe_current_scene()

    assert result["status"] == "success"
    assert result["objects"] == []
    assert result["relationships"] == []
    assert result["graph"]["nodes"] == []
    assert result["memory"]["visible_objects"] == []
    assert scene["scene_description"]["description"] == "No objects are currently visible."
    assert scene["speech_response"]["spoken_text"] == "No objects are currently visible."


def test_depth_shape_warning_propagates_without_depth_relations() -> None:
    pipeline = PipelineOrchestrator()
    result = process(
        pipeline,
        "bad-depth",
        1,
        [detection("bottle", (2, 2, 12, 12), track_id=7)],
        np.ones((10, 10), dtype=np.float32),
    )

    assert result["status"] == "partial"
    assert any("do not match" in warning for warning in result["warnings"])
    assert result["objects"][0]["relative_depth_value"] is None


def test_invalid_frames_do_not_advance_memory_or_replace_latest_state() -> None:
    pipeline = PipelineOrchestrator()
    process(pipeline, "valid", 1, [detection("bottle", (2, 2, 12, 12), track_id=7)])
    invalid_dimensions = pipeline.process_frame(
        frame_id="bad-dimensions",
        frame_number=2,
        image_dimensions={"width": "40", "height": 20},
        detections=[],
    )
    repeated_id = process(pipeline, "valid", 2, [])
    zero_frame_number = pipeline.process_frame(
        frame_id="zero-frame",
        frame_number=0,
        image_dimensions=(20, 40),
        detections=[],
    )
    malformed_detections = pipeline.process_frame(
        frame_id="malformed",
        frame_number=2,
        image_dimensions=(20, 40),
        detections=[None],
    )

    assert invalid_dimensions["status"] == "invalid_input"
    assert repeated_id["status"] == "invalid_input"
    assert zero_frame_number["status"] == "invalid_input"
    assert malformed_detections["status"] == "invalid_input"
    assert pipeline.query("Is the bottle currently visible?")["status"] == "visible"
    assert pipeline.scene_memory.get_object(7)["last_seen_frame"] == 1


def test_downstream_exception_after_memory_update_preserves_committed_scene(monkeypatch) -> None:
    pipeline = PipelineOrchestrator()
    process(pipeline, "committed", 1, [detection("bottle", (2, 2, 12, 12), track_id=7)])
    original_memory = pipeline.scene_memory.get_object(7)
    original_query = pipeline.query("Is the bottle currently visible?")
    original_graph = pipeline._latest_graph
    original_spatial = pipeline._latest_spatial

    def fail_description(*args, **kwargs):
        raise RuntimeError("synthetic description failure")

    monkeypatch.setattr(pipeline_module, "describe_scene", fail_description)
    failed = process(pipeline, "failed", 2, [detection("chair", (20, 2, 30, 12), track_id=9)])

    assert failed["status"] == "error"
    assert pipeline.scene_memory.get_object(7) == original_memory
    assert pipeline.scene_memory.get_object(9) is None
    assert pipeline.scene_memory._frame_number == 1
    assert pipeline._latest_spatial is original_spatial
    assert pipeline._latest_graph is original_graph
    assert pipeline._latest_frame_id == "committed"
    assert pipeline._latest_frame_number == 1
    assert pipeline._seen_frame_ids == {"committed"}
    assert pipeline.query("Is the bottle currently visible?") == original_query

    monkeypatch.undo()
    recovered = process(pipeline, "recovered", 2, [detection("chair", (20, 2, 30, 12), track_id=9)])

    assert recovered["status"] in {"success", "partial"}
    assert recovered["frame"]["frame_number"] == 2
    assert recovered["objects"][0]["class_name"] == "chair"
    assert pipeline._latest_frame_id == "recovered"
    assert pipeline.scene_memory.get_object(9)["last_seen_frame"] == 2


def test_query_before_first_frame_is_not_ready() -> None:
    pipeline = PipelineOrchestrator()

    result = pipeline.query("Where is the bottle?")

    assert result["status"] == "not_ready"
    assert result["query_result"] is None
    assert result["warnings"] == ["no valid frame has been processed"]