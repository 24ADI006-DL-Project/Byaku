import pytest

from tracking.tracker import ObjectTracker


def detection(class_name, box, confidence=0.9):
    return {
        "class_name": class_name,
        "confidence": confidence,
        "bounding_box": box,
    }


def test_tracker_initializes():
    tracker = ObjectTracker()

    assert tracker is not None


def test_tracker_generates_persistent_id_across_frames():
    tracker = ObjectTracker()
    first = tracker.update([detection("person", [10, 10, 50, 90])])
    second = tracker.update([detection("person", [11, 10, 51, 90])])

    assert len(first) == 1
    assert len(second) == 1
    assert first[0]["track_id"] > 0
    assert second[0]["track_id"] == first[0]["track_id"]


def test_tracker_returns_required_output_structure():
    tracker = ObjectTracker()
    tracks = tracker.update([detection("bottle", [20, 25, 45, 80], 0.86)])

    assert len(tracks) == 1
    assert set(tracks[0]) == {"track_id", "class_name", "confidence", "bounding_box"}
    assert tracks[0]["class_name"] == "bottle"
    assert tracks[0]["confidence"] == pytest.approx(0.86)
    assert len(tracks[0]["bounding_box"]) == 4


def test_tracker_handles_multiple_objects():
    tracker = ObjectTracker()
    tracks = tracker.update(
        [
            detection("person", [10, 10, 50, 90]),
            detection("chair", [100, 20, 160, 100]),
        ]
    )

    assert len(tracks) == 2
    assert {track["class_name"] for track in tracks} == {"person", "chair"}
    assert len({track["track_id"] for track in tracks}) == 2


def test_tracker_handles_empty_frame_and_continues_after_occlusion():
    tracker = ObjectTracker(track_buffer=5)
    first = tracker.update([detection("person", [10, 10, 50, 90])])
    missing = tracker.update([])
    returned = tracker.update([detection("person", [11, 10, 51, 90])])

    assert missing == []
    assert len(returned) == 1
    assert returned[0]["track_id"] == first[0]["track_id"]


def test_tracker_assigns_new_id_when_object_enters():
    tracker = ObjectTracker()
    first = tracker.update([detection("person", [10, 10, 50, 90])])
    next_frame = tracker.update(
        [
            detection("person", [11, 10, 51, 90]),
            detection("bottle", [200, 30, 230, 80]),
        ]
    )
    confirmed_frame = tracker.update(
        [
            detection("person", [12, 10, 52, 90]),
            detection("bottle", [201, 30, 231, 80]),
        ]
    )

    person = next(track for track in confirmed_frame if track["class_name"] == "person")
    bottle = next(track for track in confirmed_frame if track["class_name"] == "bottle")
    assert person["track_id"] == first[0]["track_id"]
    assert bottle["track_id"] != person["track_id"]


def test_tracker_does_not_return_object_after_it_leaves():
    tracker = ObjectTracker()
    tracker.update([detection("person", [10, 10, 50, 90])])

    assert tracker.update([]) == []
    assert tracker.update([]) == []


def test_tracker_ignores_malformed_detections():
    tracker = ObjectTracker()

    assert tracker.update([{"class_name": "person"}]) == []
