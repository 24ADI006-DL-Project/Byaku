from reasoning.events import EventDetector, EventThresholds
from reasoning.priority import PriorityConfig, PriorityScorer


def spatial_object(
    track_id=1,
    class_name="bottle",
    distance_category="moderate",
    horizontal_position="center",
    relative_depth=0.5,
):
    return {
        "track_id": track_id,
        "class_name": class_name,
        "distance_category": distance_category,
        "horizontal_position": horizontal_position,
        "relative_depth": relative_depth,
    }


def detector(**thresholds):
    return EventDetector(
        thresholds=EventThresholds(**thresholds),
        priority_scorer=PriorityScorer(PriorityConfig(recently_announced_penalty=0)),
    )


def test_new_object_event():
    events = detector().update([spatial_object(distance_category="far")])

    assert len(events) == 1
    assert events[0]["event_type"] == "new_object"
    assert events[0]["track_id"] == 1
    assert events[0]["class_name"] == "bottle"
    assert events[0]["message_required"] is False


def test_object_approaching_event_for_far_to_close():
    event_detector = detector()
    event_detector.update([spatial_object(distance_category="far")])

    events = event_detector.update([spatial_object(distance_category="close")])

    approaching = next(event for event in events if event["event_type"] == "object_approaching")
    assert approaching["previous_distance_category"] == "far"
    assert approaching["distance_category"] == "close"
    assert approaching["message_required"] is True


def test_distance_category_change_event_for_moving_farther():
    event_detector = detector()
    event_detector.update([spatial_object(distance_category="close")])

    events = event_detector.update([spatial_object(distance_category="moderate")])

    assert any(event["event_type"] == "distance_category_changed" for event in events)


def test_position_change_event():
    event_detector = detector()
    event_detector.update([spatial_object(horizontal_position="left")])

    events = event_detector.update([spatial_object(horizontal_position="right")])

    position_event = next(event for event in events if event["event_type"] == "position_changed")
    assert position_event["previous_position"] == "left"
    assert position_event["horizontal_position"] == "right"


def test_disappearance_waits_for_configured_missing_frames():
    event_detector = detector(disappearance_frames=2)
    event_detector.update([spatial_object()])

    assert event_detector.update([]) == []
    events = event_detector.update([])

    assert len(events) == 1
    assert events[0]["event_type"] == "object_disappeared"


def test_cooldown_suppresses_repeated_event_type():
    event_detector = detector(cooldown_frames=4, disappearance_frames=1)
    event_detector.update([spatial_object()])
    event_detector.update([])
    event_detector.update([spatial_object()])

    assert event_detector.update([]) == []


def test_duplicate_new_object_event_is_suppressed():
    event_detector = detector(cooldown_frames=5, disappearance_frames=1)
    event_detector.update([spatial_object()])
    event_detector.update([])

    assert event_detector.update([spatial_object()]) == []


def test_unchanged_object_emits_no_event():
    event_detector = detector()
    obj = spatial_object()
    event_detector.update([obj])

    assert event_detector.update([obj]) == []


def test_significant_continuous_depth_change_emits_spatial_change():
    event_detector = detector(significant_depth_delta=0.2)
    event_detector.update([spatial_object(distance_category="moderate", relative_depth=0.3)])

    events = event_detector.update(
        [spatial_object(distance_category="moderate", relative_depth=0.6)]
    )

    assert any(event["event_type"] == "significant_spatial_change" for event in events)