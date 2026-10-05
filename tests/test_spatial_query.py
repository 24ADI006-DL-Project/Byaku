from scene_graph.memory import SceneMemory
from query.spatial_query import SpatialQueryEngine


def spatial_object(track_id, class_name, position="left", category="close"):
    return {
        "track_id": track_id,
        "class_name": class_name,
        "confidence": 0.9,
        "bounding_box": [10, 10, 30, 40],
        "relative_depth": 0.3,
        "horizontal_position": position,
        "distance_category": category,
    }


class FakeNodes:
    def __init__(self, nodes):
        self._nodes = nodes

    def __call__(self, data=False):
        if data:
            return [(track_id, dict(attributes)) for track_id, attributes in self._nodes.items()]
        return list(self._nodes)


class FakeGraph:
    def __init__(self, nodes=None):
        self.nodes = FakeNodes(nodes or {})


class FakeSceneGraph:
    def __init__(self, objects=()):
        self.graph = FakeGraph({obj["track_id"]: obj for obj in objects})


def test_returns_matching_object_from_scene_graph():
    scene = FakeSceneGraph([spatial_object(3, "bottle")])

    result = SpatialQueryEngine(scene).answer("Where is my bottle?")

    assert result["found"] is True
    assert result["target"] == "bottle"
    assert result["objects"][0]["track_id"] == 3


def test_reports_object_not_found():
    result = SpatialQueryEngine(FakeSceneGraph()).answer("Is there a bottle?")

    assert result["found"] is False
    assert result["target"] == "bottle"
    assert result["status"] == "not_found"


def test_returns_all_objects_when_class_has_multiple_tracks():
    scene = FakeSceneGraph(
        [spatial_object(1, "chair", "left"), spatial_object(2, "chair", "right")]
    )

    result = SpatialQueryEngine(scene).answer("Where is the chair?")

    assert result["found"] is True
    assert [obj["track_id"] for obj in result["objects"]] == [1, 2]


def test_retrieves_precomputed_spatial_fields_without_recalculation():
    obj = spatial_object(5, "laptop", position="center", category="moderate")
    result = SpatialQueryEngine(FakeSceneGraph([obj])).answer("Where is the laptop?")

    assert result["objects"] == [obj]
    assert result["objects"][0]["horizontal_position"] == "center"
    assert result["objects"][0]["distance_category"] == "moderate"


def test_empty_scene_graph_returns_not_found():
    result = SpatialQueryEngine(FakeSceneGraph()).answer("Where is the person?")

    assert result["found"] is False
    assert result["objects"] == []


def test_can_search_scene_memory():
    memory = SceneMemory()
    memory.update([spatial_object(8, "bottle")])

    result = SpatialQueryEngine(memory).answer("Is there a bottle?")

    assert result["found"] is True
    assert result["objects"][0]["track_id"] == 8


def test_unknown_object_query_returns_not_found_without_invention():
    result = SpatialQueryEngine(FakeSceneGraph()).answer("Where is my toaster?")

    assert result["found"] is False
    assert result["target"] == "toaster"
    assert result["status"] == "not_found"