import networkx as nx

from scene_graph.scene_graph import SceneGraph


def spatial_object(track_id, class_name="bottle", **updates):
    return {
        "track_id": track_id,
        "class_name": class_name,
        "confidence": 0.9,
        "bounding_box": [10, 10, 30, 30],
        "relative_depth": 0.3,
        "horizontal_position": "left",
        "distance_category": "close",
        **updates,
    }


def test_creates_networkx_scene_graph():
    scene = SceneGraph()

    assert isinstance(scene.graph, nx.MultiDiGraph)
    assert len(scene.graph) == 0


def test_adds_object_using_track_id_and_stores_spatial_data():
    scene = SceneGraph()
    obj = spatial_object(3)

    scene.add_object(obj)

    assert scene.graph.nodes[3]["class_name"] == "bottle"
    assert scene.graph.nodes[3]["track_id"] == 3
    assert scene.graph.nodes[3]["distance_category"] == "close"


def test_updates_existing_object_without_duplicate_node():
    scene = SceneGraph()
    scene.add_object(spatial_object(3, confidence=0.6))
    scene.update_object(spatial_object(3, confidence=0.95, horizontal_position="center"))

    assert list(scene.graph.nodes) == [3]
    assert scene.graph.nodes[3]["confidence"] == 0.95
    assert scene.graph.nodes[3]["horizontal_position"] == "center"


def test_removes_object_by_track_id():
    scene = SceneGraph()
    scene.add_object(spatial_object(3))

    assert scene.remove_object(3) is True
    assert scene.remove_object(3) is False
    assert 3 not in scene.graph


def test_adds_multiple_objects():
    scene = SceneGraph()
    scene.update_objects([spatial_object(1), spatial_object(2, "chair")])

    assert set(scene.graph.nodes) == {1, 2}