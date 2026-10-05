from byaku_rebuild.scene_memory import SceneMemory


def make_object(
    track_id: object = 7,
    *,
    class_name: str = "bottle",
    box: tuple[float, float, float, float] = (10, 20, 30, 40),
    center: tuple[float, float] | None = (20, 30),
    depth: float | None = 0.75,
    confidence: float | None = None,
) -> dict:
    observation = {
        "track_id": track_id,
        "class_name": class_name,
        "bounding_box": box,
    }
    if center is not None:
        observation["center"] = center
    if depth is not None:
        observation["relative_depth_value"] = depth
    if confidence is not None:
        observation["confidence"] = confidence
    return observation


def test_adds_new_objects_with_spatial_attributes_and_last_seen() -> None:
    memory = SceneMemory()
    result = memory.update([make_object()], frame_number=1, timestamp=100.5)

    item = result["visible_objects"][0]
    assert item["object_id"] == "track:7"
    assert item["track_id"] == 7
    assert item["class_name"] == "bottle"
    assert item["bounding_box"] == (10.0, 20.0, 30.0, 40.0)
    assert item["center"] == (20.0, 30.0)
    assert item["relative_depth_value"] == 0.75
    assert item["last_seen_frame"] == 1
    assert item["last_seen_timestamp"] == 100.5
    assert item["status"] == "visible"


def test_updates_same_tracked_object_and_appends_observation() -> None:
    memory = SceneMemory()
    memory.update([make_object(box=(10, 20, 30, 40))], frame_number=1)
    result = memory.update(
        [make_object(box=(20, 25, 40, 45), center=(30, 35), depth=0.6)],
        frame_number=2,
    )

    item = result["visible_objects"][0]
    assert len(result["visible_objects"]) == 1
    assert item["object_id"] == "track:7"
    assert item["bounding_box"] == (20.0, 25.0, 40.0, 45.0)
    assert item["center"] == (30.0, 35.0)
    assert item["relative_depth_value"] == 0.6
    assert item["last_seen_frame"] == 2
    assert [observation["frame_number"] for observation in item["observations"]] == [1, 2]


def test_tracks_multiple_objects_in_a_frame() -> None:
    memory = SceneMemory()
    result = memory.update([make_object(1), make_object(2, class_name="chair")], frame_number=1)

    assert {item["object_id"] for item in result["visible_objects"]} == {"track:1", "track:2"}
    assert len(result["temporarily_missing_objects"]) == 0


def test_disappearance_is_temporary_and_reappearance_restores_visibility() -> None:
    memory = SceneMemory(stale_after_frames=4)
    memory.update([make_object()], frame_number=1)
    missing = memory.update([], frame_number=2)

    assert missing["visible_objects"] == []
    assert [item["object_id"] for item in missing["temporarily_missing_objects"]] == ["track:7"]
    assert missing["stale_objects"] == []

    returned = memory.update([make_object(box=(12, 20, 32, 40))], frame_number=3)
    assert [item["object_id"] for item in returned["visible_objects"]] == ["track:7"]
    assert returned["temporarily_missing_objects"] == []
    assert returned["visible_objects"][0]["last_seen_frame"] == 3
    assert len(memory.get_recent_observations(7)) == 2


def test_object_becomes_stale_after_configured_frame_gap_not_visible() -> None:
    memory = SceneMemory(stale_after_frames=3)
    memory.update([make_object()], frame_number=1)
    memory.update([], frame_number=2)
    still_missing = memory.update([], frame_number=3)

    assert [item["object_id"] for item in still_missing["temporarily_missing_objects"]] == ["track:7"]
    assert still_missing["visible_objects"] == []

    stale = memory.update([], frame_number=4)
    assert [item["object_id"] for item in stale["stale_objects"]] == ["track:7"]
    assert stale["visible_objects"] == []
    assert stale["temporarily_missing_objects"] == []
    assert memory.get_object(7)["status"] == "stale"


def test_missing_depth_and_empty_frames_are_safe() -> None:
    memory = SceneMemory()
    result = memory.update([make_object(depth=None)], frame_number=1)

    assert result["visible_objects"][0]["relative_depth_value"] is None
    assert memory.update([], frame_number=2)["visible_objects"] == []
    assert memory.get_object(7)["last_seen_frame"] == 1


def test_invalid_ids_and_boxes_are_handled_without_false_tracking() -> None:
    memory = SceneMemory()
    result = memory.update(
        [
            make_object([], box=(1, 1, 5, 5)),
            make_object(3, box=(5, 5, 5, 9)),
            make_object(3, box=(1, 1, 9, 9)),
            make_object(3, class_name="chair", box=(2, 2, 10, 10)),
        ],
        frame_number=1,
    )

    visible = result["visible_objects"]
    assert len(visible) == 2
    assert {item["object_id"] for item in visible} == {"temporary:1:0", "track:3"}
    assert next(item for item in visible if item["object_id"] == "track:3")["class_name"] == "bottle"
    assert any("invalid tracking ID" in warning for warning in result["warnings"])
    assert any("invalid bounding box" in warning for warning in result["warnings"])
    assert any("duplicate tracking ID" in warning for warning in result["warnings"])


def test_temporary_ids_are_not_reused_across_frames() -> None:
    memory = SceneMemory()
    first = memory.update(
        [{"object_id": "frame:0", "track_id": None, "class_name": "person", "bounding_box": (0, 0, 5, 5)}],
        frame_number=1,
    )
    second = memory.update(
        [{"object_id": "frame:0", "track_id": None, "class_name": "person", "bounding_box": (1, 0, 6, 5)}],
        frame_number=2,
    )

    assert first["visible_objects"][0]["object_id"] == "temporary:1:0"
    assert second["visible_objects"][0]["object_id"] == "temporary:2:0"
    assert second["temporarily_missing_objects"] == []
    assert len(memory.get_visible_objects()) == 1


def test_recent_observation_query_returns_bounded_ordered_history() -> None:
    memory = SceneMemory(history_size=3)
    for frame_number in range(1, 5):
        memory.update([make_object(box=(frame_number, 0, frame_number + 5, 5))], frame_number=frame_number)

    observations = memory.get_recent_observations("track:7", limit=2)
    assert [item["frame_number"] for item in observations] == [3, 4]
    assert len(memory.get_recent_observations(7)) == 3
    assert memory.get_recent_observations(7, limit=0) == []
    assert memory.get_recent_observations(99) == []


def test_malformed_observations_and_centers_are_skipped_or_repaired() -> None:
    memory = SceneMemory()
    result = memory.update(
        [
            None,
            {"track_id": 1, "class_name": "", "bounding_box": (0, 0, 5, 5)},
            {"track_id": 2, "class_name": "chair", "bounding_box": (0, 0, 6, 8), "center": 4},
        ]
    )

    assert len(result["visible_objects"]) == 1
    assert result["visible_objects"][0]["center"] == (3.0, 4.0)
    assert len(result["warnings"]) == 3


def test_reused_frame_number_advances_monotonically_with_warning() -> None:
    memory = SceneMemory()
    first = memory.update([], frame_number=4)
    second = memory.update([], frame_number=4)

    assert first["frame_number"] == 4
    assert second["frame_number"] == 5
    assert any("advanced automatically" in warning for warning in second["warnings"])


def test_spatial_temporary_id_is_preserved_for_its_frame_and_reuse_is_remapped() -> None:
    memory = SceneMemory()
    first = memory.update(
        [{"object_id": "temporary:source-1:0", "class_name": "bottle", "bounding_box": (0, 0, 5, 5)}],
        frame_number=1,
    )
    second = memory.update(
        [{"object_id": "temporary:source-1:0", "class_name": "bottle", "bounding_box": (1, 0, 6, 5)}],
        frame_number=2,
    )

    assert first["visible_objects"][0]["object_id"] == "temporary:source-1:0"
    assert second["visible_objects"][0]["object_id"] == "temporary:2:0"
    assert any("reused across frames" in warning for warning in second["warnings"])
    assert memory.get_object("temporary:source-1:0") is None


def test_confidence_is_retained_when_supplied_and_removed_when_omitted() -> None:
    memory = SceneMemory()
    first = memory.update([make_object(confidence=0.73)], frame_number=1)
    second = memory.update([make_object(depth=0.6)], frame_number=2)

    assert first["visible_objects"][0]["confidence"] == 0.73
    assert first["visible_objects"][0]["observations"][0]["confidence"] == 0.73
    assert "confidence" not in second["visible_objects"][0]
    assert "confidence" not in second["visible_objects"][0]["observations"][-1]
    assert second["visible_objects"][0]["observations"][0]["confidence"] == 0.73


def test_stale_retention_keeps_records_for_reappearance_then_evicts_after_limit() -> None:
    memory = SceneMemory(stale_after_frames=2, history_size=3, stale_retention_frames=4)
    memory.update([make_object(box=(1, 1, 8, 8))], frame_number=1)
    memory.update([], frame_number=2)
    stale = memory.update([], frame_number=3)

    assert [item["object_id"] for item in stale["stale_objects"]] == ["track:7"]
    returned = memory.update([make_object(box=(2, 1, 9, 8))], frame_number=4)
    assert returned["visible_objects"][0]["status"] == "visible"
    assert len(memory.get_recent_observations(7)) == 2

    memory.update([], frame_number=5)
    memory.update([], frame_number=6)
    retained_at_limit = memory.update([], frame_number=8)
    assert [item["object_id"] for item in retained_at_limit["stale_objects"]] == ["track:7"]
    assert len(memory.get_recent_observations(7)) == 2

    evicted = memory.update([], frame_number=9)
    assert evicted["stale_objects"] == []
    assert memory.get_object(7) is None
    assert memory.get_recent_observations(7) == []


def test_stale_retention_limit_must_allow_stale_transition() -> None:
    import pytest

    with pytest.raises(ValueError):
        SceneMemory(stale_after_frames=5, stale_retention_frames=4)