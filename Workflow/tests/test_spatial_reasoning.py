"""Step 26 — Spatial Reasoning Validation Tests.

Validates the accuracy and consistency of BYAKU's spatial understanding:
1. Object positions from camera perspective (left, center, right, upper, middle, lower).
2. Relative depth consistency, handling unavailable depth, and preventing metric/safety claims.
3. Tracking consistency, stable continuation, distinct object separation, and duplicate handling.
4. Scene graph and memory across consecutive frames (moving objects, stale transitions, reappearance).
5. Query consistency across dynamic scenes (verifying answers agree with latest frame only).
"""

from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pytest

from Workflow.vision.object_detector_tracker import DetectionTrackingAdapter
from Workflow.scene_pipeline import PipelineOrchestrator
from Workflow.reasoning.object_query_engine import answer_query
from Workflow.reasoning.scene_relationship_graph import build_scene_graph, get_objects
from Workflow.reasoning.scene_memory import SceneMemory
from Workflow.reasoning.spatial_relation_engine import (
    bounding_box_iou,
    calculate_spatial_relationships,
)


WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
VIDEO_PATH = WORKSPACE_ROOT / "Data" / "samples" / "indoor_objects_video.mp4"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_det(
    box: list[float],
    class_name: str = "object",
    track_id: int | None = None,
    confidence: float = 0.9,
) -> dict[str, Any]:
    det = {
        "class_name": class_name,
        "bounding_box": [float(x) for x in box],
        "confidence": confidence,
    }
    if track_id is not None:
        det["track_id"] = track_id
    return det


# ---------------------------------------------------------------------------
# 1. Object positions from camera perspective
# ---------------------------------------------------------------------------


def test_camera_perspective_horizontal_positions() -> None:
    """Objects are classified as left, center, or right from camera perspective."""
    width, height = 300, 200
    # center zone fraction 0.2 means:
    # left: center_x < 300 * 0.3 = 90
    # center: 90 <= center_x <= 210
    # right: center_x > 300 * 0.7 = 210
    dets = [
        make_det([10, 80, 50, 120], "cup", track_id=1),       # center_x = 30 -> left
        make_det([130, 80, 170, 120], "bottle", track_id=2),  # center_x = 150 -> center
        make_det([250, 80, 290, 120], "chair", track_id=3),   # center_x = 270 -> right
    ]
    spatial = calculate_spatial_relationships(dets, (height, width), center_zone_fraction=0.2)
    objs = {obj["track_id"]: obj for obj in spatial["objects"]}

    assert objs[1]["horizontal_position"] == "left"
    assert objs[2]["horizontal_position"] == "center"
    assert objs[3]["horizontal_position"] == "right"


def test_camera_perspective_vertical_positions() -> None:
    """Vertical relationships are classified as above, middle, or below."""
    width, height = 200, 300
    # center zone fraction 0.2 means:
    # above: center_y < 300 * 0.3 = 90
    # middle: 90 <= center_y <= 210
    # below: center_y > 300 * 0.7 = 210
    dets = [
        make_det([80, 10, 120, 50], "clock", track_id=1),     # center_y = 30 -> above
        make_det([80, 130, 120, 170], "tv", track_id=2),      # center_y = 150 -> middle
        make_det([80, 250, 120, 290], "rug", track_id=3),     # center_y = 270 -> below
    ]
    spatial = calculate_spatial_relationships(dets, (height, width), center_zone_fraction=0.2)
    objs = {obj["track_id"]: obj for obj in spatial["objects"]}

    assert objs[1]["vertical_position"] == "above"
    assert objs[2]["vertical_position"] == "middle"
    assert objs[3]["vertical_position"] == "below"


def test_moving_object_updates_horizontal_position() -> None:
    """When an object moves across consecutive frames, its position updates."""
    orchestrator = PipelineOrchestrator()
    # Frame 1: person is on the left
    f1 = orchestrator.process_frame(
        frame_id="f1",
        frame_number=1,
        image_dimensions={"width": 300, "height": 200},
        detections=[make_det([10, 50, 50, 150], "person", track_id=1)],
    )
    assert f1["objects"][0]["horizontal_position"] == "left"
    q1 = orchestrator.query("Where is the person?")
    assert "on the left" in q1["query_result"]["answer"]

    # Frame 2: person moves to center
    f2 = orchestrator.process_frame(
        frame_id="f2",
        frame_number=2,
        image_dimensions={"width": 300, "height": 200},
        detections=[make_det([130, 50, 170, 150], "person", track_id=1)],
    )
    assert f2["objects"][0]["horizontal_position"] == "center"
    q2 = orchestrator.query("Where is the person?")
    assert "near the center" in q2["query_result"]["answer"]

    # Frame 3: person moves to right
    f3 = orchestrator.process_frame(
        frame_id="f3",
        frame_number=3,
        image_dimensions={"width": 300, "height": 200},
        detections=[make_det([250, 50, 290, 150], "person", track_id=1)],
    )
    assert f3["objects"][0]["horizontal_position"] == "right"
    q3 = orchestrator.query("Where is the person?")
    assert "on the right" in q3["query_result"]["answer"]


# ---------------------------------------------------------------------------
# 2. Relative depth consistency and guardrails
# ---------------------------------------------------------------------------


def test_relative_depth_nearer_and_farther_relationships() -> None:
    """Higher depth values indicate nearer objects, generating consistent pairs."""
    height, width = 100, 100
    depth_map = np.zeros((height, width), dtype=np.float32)
    # Box 1 (foreground): high depth values ~ 0.85
    depth_map[20:60, 10:40] = 0.85
    # Box 2 (background): lower depth values ~ 0.25
    depth_map[20:60, 60:90] = 0.25

    dets = [
        make_det([10, 20, 40, 60], "bottle", track_id=1),
        make_det([60, 20, 90, 60], "chair", track_id=2),
    ]
    spatial = calculate_spatial_relationships(dets, (height, width), depth_map=depth_map)
    rel_map = {(r["subject"], r["relation"], r["object"]) for r in spatial["relationships"]}

    # bottle is nearer than chair; chair is farther than bottle
    assert ("track:1", "nearer_than", "track:2") in rel_map
    assert ("track:2", "farther_than", "track:1") in rel_map


def test_relative_depth_similar_within_threshold() -> None:
    """Objects with depth difference <= threshold are labeled similar_depth."""
    height, width = 100, 100
    depth_map = np.full((height, width), 0.50, dtype=np.float32)
    depth_map[20:60, 60:90] = 0.52  # difference 0.02 <= threshold 0.05

    dets = [
        make_det([10, 20, 40, 60], "cup", track_id=1),
        make_det([60, 20, 90, 60], "bowl", track_id=2),
    ]
    spatial = calculate_spatial_relationships(
        dets, (height, width), depth_map=depth_map, depth_difference_threshold=0.05
    )
    rel_types = [r["relation"] for r in spatial["relationships"]]
    assert "similar_depth" in rel_types
    assert "nearer_than" not in rel_types
    assert "farther_than" not in rel_types


def test_depth_guardrails_never_claims_meters_or_safety() -> None:
    """Answers and speech text must never report metric distances or safety claims."""
    orc = PipelineOrchestrator()
    height, width = 100, 100
    depth_map = np.full((height, width), 0.7, dtype=np.float32)

    orc.process_frame(
        frame_id="depth-frame-1",
        frame_number=1,
        image_dimensions={"width": width, "height": height},
        detections=[
            make_det([10, 20, 40, 60], "bottle", track_id=1),
            make_det([60, 20, 90, 60], "person", track_id=2),
        ],
        depth_map=depth_map,
    )

    for q_text in [
        "Where is the bottle?",
        "Where is the person?",
        "What is in front of the chair?",
        "What objects are visible?",
    ]:
        result = orc.query(q_text)
        answer = (result["query_result"]["answer"] or "").lower()
        spoken = ((result["speech_response"] or {}).get("spoken_text") or "").lower()

        for forbidden in ["meter", "metre", "centimeter", "feet", "inch", "safe to approach"]:
            assert forbidden not in answer, f"Forbidden term '{forbidden}' in answer: {answer}"
            assert forbidden not in spoken, f"Forbidden term '{forbidden}' in spoken text: {spoken}"


def test_unavailable_depth_handled_without_inventing_values() -> None:
    """When depth is None or invalid, relative depth is None and relations are omitted."""
    height, width = 100, 100
    dets = [
        make_det([10, 20, 40, 60], "cup", track_id=1),
        make_det([60, 20, 90, 60], "plate", track_id=2),
    ]
    # Test with None depth
    none_res = calculate_spatial_relationships(dets, (height, width), depth_map=None)
    for obj in none_res["objects"]:
        assert obj["relative_depth_value"] is None
    depth_rels = [r for r in none_res["relationships"] if r["relation"] in {"nearer_than", "farther_than", "similar_depth"}]
    assert len(depth_rels) == 0

    # Test with mismatched shape
    mismatched_depth = np.zeros((50, 50), dtype=np.float32)
    mismatch_res = calculate_spatial_relationships(dets, (height, width), depth_map=mismatched_depth)
    assert any("match" in w or "dimensions" in w for w in mismatch_res["warnings"])
    for obj in mismatch_res["objects"]:
        assert obj["relative_depth_value"] is None


# ---------------------------------------------------------------------------
# 3. Tracking consistency
# ---------------------------------------------------------------------------


def test_duplicate_detections_in_same_frame_do_not_produce_self_relations() -> None:
    """Two detections with the same track_id in one frame are handled gracefully."""
    dets = [
        make_det([10, 10, 30, 30], "bottle", track_id=5),
        make_det([50, 50, 70, 70], "bottle", track_id=5),  # duplicate track_id
    ]
    spatial = calculate_spatial_relationships(dets, (100, 100))
    # spatial calculation keeps both but does not generate self-relations (track:5 to track:5)
    for rel in spatial["relationships"]:
        assert rel["subject"] != rel["object"]

    # When fed to SceneMemory, the duplicate is dropped with a warning
    mem = SceneMemory()
    snapshot = mem.update(spatial, frame_number=1)
    visible_ids = [obj["object_id"] for obj in snapshot["visible_objects"]]
    assert visible_ids == ["track:5"]
    assert any("duplicate" in w for w in snapshot["warnings"])


def test_different_objects_not_incorrectly_merged() -> None:
    """Distinct tracks maintain independent positions, statuses, and histories."""
    mem = SceneMemory()
    mem.update(
        [
            make_det([10, 10, 30, 30], "chair", track_id=10),
            make_det([60, 60, 90, 90], "table", track_id=20),
        ],
        frame_number=1,
    )
    assert mem.get_object(10)["class_name"] == "chair"
    assert mem.get_object(20)["class_name"] == "table"

    # Frame 2: chair moves, table disappears
    mem.update(
        [
            make_det([15, 15, 35, 35], "chair", track_id=10),
        ],
        frame_number=2,
    )
    assert mem.get_object(10)["status"] == "visible"
    assert mem.get_object(10)["bounding_box"] == (15.0, 15.0, 35.0, 35.0)
    assert mem.get_object(20)["status"] == "missing"
    assert mem.get_object(20)["bounding_box"] == (60.0, 60.0, 90.0, 90.0)


@pytest.mark.skipif(not VIDEO_PATH.is_file(), reason="indoor_objects_video.mp4 not present")
def test_real_video_tracking_continuity_across_frames() -> None:
    """Run real Ultralytics ByteTrack on consecutive video frames to verify track continuity."""
    cap = cv2.VideoCapture(str(VIDEO_PATH))
    assert cap.isOpened(), "Could not open indoor_objects_video.mp4"

    adapter = DetectionTrackingAdapter()
    observed_tracks_by_frame: dict[int, dict[int, str]] = {}

    try:
        for frame_idx in range(1, 16):  # sample first 15 frames
            ret, frame = cap.read()
            if not ret:
                break
            res = adapter.process_frame(frame, frame_id=f"vid:{frame_idx}", frame_number=frame_idx)
            observed_tracks_by_frame[frame_idx] = {
                t["track_id"]: t["class_name"] for t in res["tracks"] if t.get("track_id") is not None
            }
    finally:
        cap.release()

    assert len(observed_tracks_by_frame) >= 10, "Should have processed at least 10 frames"

    # Verify that at least one track persists across adjacent frames
    continuations = 0
    for f in range(1, len(observed_tracks_by_frame)):
        prev_ids = set(observed_tracks_by_frame[f].keys())
        curr_ids = set(observed_tracks_by_frame[f + 1].keys())
        shared = prev_ids.intersection(curr_ids)
        if shared:
            continuations += 1

    assert continuations >= 8, f"Expected consistent tracking continuity, got {continuations} continuations"


# ---------------------------------------------------------------------------
# 4. Scene graph and memory consistency
# ---------------------------------------------------------------------------


def test_scene_graph_nodes_correspond_to_current_detections() -> None:
    """Scene graph nodes strictly correspond to current detections in that frame."""
    dets = [
        make_det([10, 10, 30, 30], "laptop", track_id=1),
        make_det([50, 50, 80, 80], "mouse", track_id=2),
    ]
    spatial = calculate_spatial_relationships(dets, (100, 100))
    graph = build_scene_graph(spatial)
    graph_objects = get_objects(graph)
    node_ids = {obj["object_id"] for obj in graph_objects}

    assert node_ids == {"track:1", "track:2"}


def test_scene_graph_updates_relationships_when_object_moves() -> None:
    """When object moves, old relationships are removed and new relationships reflect current state."""
    orc = PipelineOrchestrator()

    # Frame 1: bottle is left of chair
    f1 = orc.process_frame(
        frame_id="f1",
        frame_number=1,
        image_dimensions={"width": 200, "height": 100},
        detections=[
            make_det([10, 20, 30, 60], "bottle", track_id=1),
            make_det([150, 20, 180, 60], "chair", track_id=2),
        ],
    )
    g1 = f1["graph"]
    g1_edges = {(e["source"], e["relation"], e["target"]) for e in g1["edges"]}
    assert ("track:1", "left_of", "track:2") in g1_edges

    # Frame 2: bottle moves to the right of chair
    f2 = orc.process_frame(
        frame_id="f2",
        frame_number=2,
        image_dimensions={"width": 200, "height": 100},
        detections=[
            make_det([170, 20, 190, 60], "bottle", track_id=1),
            make_det([30, 20, 60, 60], "chair", track_id=2),
        ],
    )
    g2 = f2["graph"]
    g2_edges = {(e["source"], e["relation"], e["target"]) for e in g2["edges"]}

    assert ("track:1", "right_of", "track:2") in g2_edges
    assert ("track:1", "left_of", "track:2") not in g2_edges


def test_stale_objects_are_marked_and_excluded_from_current_graph() -> None:
    """Stale objects transition through missing -> stale and are not nodes in current graph."""
    orc = PipelineOrchestrator()
    orc.scene_memory.stale_after_frames = 3

    # Frame 1: bottle and chair visible
    orc.process_frame(
        frame_id="f1",
        frame_number=1,
        image_dimensions={"width": 100, "height": 100},
        detections=[
            make_det([10, 10, 30, 30], "bottle", track_id=1),
            make_det([50, 50, 70, 70], "chair", track_id=2),
        ],
    )
    assert orc.scene_memory.get_object(1)["status"] == "visible"

    # Frame 2: chair disappears (1 frame missing -> missing)
    f2 = orc.process_frame(
        frame_id="f2",
        frame_number=2,
        image_dimensions={"width": 100, "height": 100},
        detections=[make_det([10, 10, 30, 30], "bottle", track_id=1)],
    )
    assert orc.scene_memory.get_object(2)["status"] == "missing"
    # Chair is NOT in current frame's graph
    assert {n["object_id"] for n in f2["graph"]["nodes"]} == {"track:1"}

    # Frames 3 and 4: chair still missing (3 frames missing -> stale)
    orc.process_frame(
        frame_id="f3",
        frame_number=3,
        image_dimensions={"width": 100, "height": 100},
        detections=[make_det([10, 10, 30, 30], "bottle", track_id=1)],
    )
    f4 = orc.process_frame(
        frame_id="f4",
        frame_number=4,
        image_dimensions={"width": 100, "height": 100},
        detections=[make_det([10, 10, 30, 30], "bottle", track_id=1)],
    )
    assert orc.scene_memory.get_object(2)["status"] == "stale"
    assert {n["object_id"] for n in f4["graph"]["nodes"]} == {"track:1"}


def test_reappearing_object_restores_visibility_and_updates_graph() -> None:
    """When a stale object reappears, its status restores to visible with new position."""
    orc = PipelineOrchestrator()
    orc.scene_memory.stale_after_frames = 2

    orc.process_frame(
        frame_id="f1",
        frame_number=1,
        image_dimensions={"width": 100, "height": 100},
        detections=[make_det([10, 10, 20, 20], "cup", track_id=7)],
    )
    # Becomes missing then stale
    orc.process_frame(
        frame_id="f2",
        frame_number=2,
        image_dimensions={"width": 100, "height": 100},
        detections=[],
    )
    orc.process_frame(
        frame_id="f3",
        frame_number=3,
        image_dimensions={"width": 100, "height": 100},
        detections=[],
    )
    assert orc.scene_memory.get_object(7)["status"] == "stale"

    # Reappears at frame 4 at a new position
    f4 = orc.process_frame(
        frame_id="f4",
        frame_number=4,
        image_dimensions={"width": 100, "height": 100},
        detections=[make_det([70, 70, 85, 85], "cup", track_id=7)],
    )
    obj = orc.scene_memory.get_object(7)
    assert obj["status"] == "visible"
    assert obj["bounding_box"] == (70.0, 70.0, 85.0, 85.0)
    assert {n["object_id"] for n in f4["graph"]["nodes"]} == {"track:7"}


# ---------------------------------------------------------------------------
# 5. Query consistency across dynamic scenes
# ---------------------------------------------------------------------------


def test_query_what_is_to_the_left_of_chair_updates_with_scene() -> None:
    """'What is to the left of the chair?' answers correctly when scene changes."""
    orc = PipelineOrchestrator()

    # Frame 1: bottle is to the left of chair
    orc.process_frame(
        frame_id="f1",
        frame_number=1,
        image_dimensions={"width": 300, "height": 100},
        detections=[
            make_det([20, 20, 40, 60], "bottle", track_id=1),
            make_det([150, 20, 190, 60], "chair", track_id=2),
        ],
    )
    q1 = orc.query("What is to the left of the chair?")
    assert q1["status"] == "visible"
    assert "bottle" in q1["query_result"]["answer"].lower()

    # Frame 2: bottle moved to the right of chair; nothing is to the left of chair
    orc.process_frame(
        frame_id="f2",
        frame_number=2,
        image_dimensions={"width": 300, "height": 100},
        detections=[
            make_det([220, 20, 250, 60], "bottle", track_id=1),
            make_det([150, 20, 190, 60], "chair", track_id=2),
        ],
    )
    q2 = orc.query("What is to the left of the chair?")
    assert q2["status"] == "not_found"
    assert "no visible object" in q2["query_result"]["answer"].lower()

    # Querying right of chair should now find the bottle
    q2_right = orc.query("What is to the right of the chair?")
    assert q2_right["status"] == "visible"
    assert "bottle" in q2_right["query_result"]["answer"].lower()


def test_query_where_is_bottle_reports_temporarily_missing_truthfully() -> None:
    """When an object disappears, query reports it as missing with last-seen info, not current."""
    orc = PipelineOrchestrator()

    # Frame 1: bottle visible in center
    orc.process_frame(
        frame_id="f1",
        frame_number=1,
        image_dimensions={"width": 200, "height": 200},
        detections=[make_det([80, 80, 120, 120], "bottle", track_id=1)],
    )
    q1 = orc.query("Where is the bottle?")
    assert q1["status"] == "visible"
    assert "currently visible" in q1["query_result"]["answer"]

    # Frame 2: bottle missing
    orc.process_frame(
        frame_id="f2",
        frame_number=2,
        image_dimensions={"width": 200, "height": 200},
        detections=[make_det([10, 10, 30, 30], "person", track_id=2)],
    )
    q2 = orc.query("Where is the bottle?")
    assert q2["status"] == "missing"
    ans2 = q2["query_result"]["answer"]
    assert "temporarily missing" in ans2
    assert "last seen" in ans2
    assert "currently visible" not in ans2


def test_query_what_objects_are_visible_excludes_stale_and_missing() -> None:
    """'What objects are visible?' lists only currently visible items."""
    orc = PipelineOrchestrator()

    orc.process_frame(
        frame_id="f1",
        frame_number=1,
        image_dimensions={"width": 200, "height": 200},
        detections=[
            make_det([20, 20, 40, 60], "bottle", track_id=1),
            make_det([150, 20, 180, 60], "chair", track_id=2),
        ],
    )
    # Frame 2: chair disappeared
    orc.process_frame(
        frame_id="f2",
        frame_number=2,
        image_dimensions={"width": 200, "height": 200},
        detections=[make_det([20, 20, 40, 60], "bottle", track_id=1)],
    )
    q = orc.query("What objects are visible?")
    assert q["status"] == "visible"
    matched_classes = {m["class_name"] for m in q["query_result"]["matched_objects"]}
    assert matched_classes == {"bottle"}
    assert "chair" not in q["query_result"]["answer"].lower()
