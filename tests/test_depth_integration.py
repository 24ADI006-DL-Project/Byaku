import numpy as np

from spatial.depth_integration import DepthIntegrator


def tracked_object(track_id, class_name, box, confidence=0.9):
    return {
        "track_id": track_id,
        "class_name": class_name,
        "confidence": confidence,
        "bounding_box": box,
    }


def test_integrates_object_with_median_depth():
    objects = [tracked_object(2, "bottle", [1, 1, 4, 3], 0.87)]
    depth_map = np.array(
        [
            [1, 1, 1, 1, 1],
            [1, 2, 4, 100, 1],
            [1, 6, 8, 200, 1],
            [1, 1, 1, 1, 1],
        ],
        dtype=np.float32,
    )

    enriched = DepthIntegrator().integrate(objects, depth_map)

    assert enriched[0]["relative_depth"] == 7.0
    assert enriched[0]["track_id"] == 2
    assert enriched[0]["class_name"] == "bottle"


def test_integrates_multiple_objects_independently():
    objects = [
        tracked_object(1, "person", [0, 0, 2, 2]),
        tracked_object(2, "chair", [2, 0, 4, 2]),
    ]
    depth_map = np.array([[1, 3, 10, 14], [5, 7, 18, 22]], dtype=np.float32)

    enriched = DepthIntegrator().integrate(objects, depth_map)

    assert [item["relative_depth"] for item in enriched] == [4.0, 16.0]
    assert [item["track_id"] for item in enriched] == [1, 2]


def test_clips_partially_outside_bounding_box():
    objects = [tracked_object(3, "chair", [-2, -1, 2, 2])]
    depth_map = np.array([[2, 4], [6, 8]], dtype=np.float32)

    enriched = DepthIntegrator().integrate(objects, depth_map)

    assert enriched[0]["relative_depth"] == 5.0


def test_invalid_bounding_box_has_no_depth():
    objects = [
        tracked_object(4, "table", [3, 1, 1, 4]),
        tracked_object(5, "person", [0, 0, float("nan"), 2]),
        {"track_id": 6, "class_name": "door", "confidence": 0.7},
    ]

    enriched = DepthIntegrator().integrate(objects, np.ones((5, 5), dtype=np.float32))

    assert [item["relative_depth"] for item in enriched] == [None, None, None]


def test_box_with_no_overlap_has_no_depth():
    objects = [tracked_object(7, "bottle", [10, 10, 12, 12])]

    enriched = DepthIntegrator().integrate(objects, np.ones((5, 5), dtype=np.float32))

    assert enriched[0]["relative_depth"] is None


def test_depth_map_image_dimension_mismatch_is_rejected():
    objects = [tracked_object(8, "person", [0, 0, 3, 3])]
    depth_map = np.ones((3, 4), dtype=np.float32)

    enriched = DepthIntegrator().integrate(objects, depth_map, image_shape=(4, 4))

    assert enriched[0]["relative_depth"] is None


def test_empty_object_list_returns_empty_list():
    assert DepthIntegrator().integrate([], np.ones((2, 2), dtype=np.float32)) == []


def test_preserves_tracking_fields_without_mutating_input():
    original = tracked_object(42, "laptop", [0, 0, 2, 2], 0.73)
    enriched = DepthIntegrator().integrate([original], np.array([[1, 3], [5, 7]], dtype=np.float32))

    assert enriched[0] == {**original, "relative_depth": 4.0}
    assert "relative_depth" not in original
