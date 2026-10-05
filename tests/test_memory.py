from scene_graph.memory import SceneMemory


def spatial_object(track_id, confidence=0.9, class_name="person"):
    return {
        "track_id": track_id,
        "class_name": class_name,
        "confidence": confidence,
        "bounding_box": [10, 10, 30, 40],
        "relative_depth": 0.3,
        "horizontal_position": "center",
        "distance_category": "close",
    }


def test_object_persists_during_short_absence():
    memory = SceneMemory(max_missing_frames=3)
    memory.update([spatial_object(1)])

    current = memory.update([])

    assert len(current) == 1
    assert current[0]["track_id"] == 1
    assert 1 in memory.scene_graph.graph


def test_existing_object_is_updated_by_track_id():
    memory = SceneMemory()
    memory.update([spatial_object(1, confidence=0.6)])

    current = memory.update([spatial_object(1, confidence=0.95, class_name="pedestrian")])

    assert len(current) == 1
    assert current[0]["confidence"] == 0.95
    assert current[0]["class_name"] == "pedestrian"
    assert memory.scene_graph.graph.nodes[1]["confidence"] == 0.95


def test_unobserved_object_expires_after_configured_frames():
    memory = SceneMemory(max_missing_frames=2)
    memory.update([spatial_object(1)])
    memory.update([])

    assert len(memory.objects) == 1
    memory.update([])

    assert memory.objects == []
    assert 1 not in memory.scene_graph.graph


def test_new_object_enters_scene_memory():
    memory = SceneMemory()

    current = memory.update([spatial_object(5, class_name="bottle")])

    assert len(current) == 1
    assert current[0]["track_id"] == 5


def test_memory_tracks_multiple_objects_and_expires_independently():
    memory = SceneMemory(max_missing_frames=2)
    memory.update([spatial_object(1), spatial_object(2, class_name="chair")])
    memory.update([spatial_object(1)])
    current = memory.update([spatial_object(1)])

    assert [obj["track_id"] for obj in current] == [1]
    assert 1 in memory.scene_graph.graph
    assert 2 not in memory.scene_graph.graph


def test_empty_memory_handles_empty_frames():
    memory = SceneMemory()

    assert memory.update([]) == []
    assert memory.objects == []