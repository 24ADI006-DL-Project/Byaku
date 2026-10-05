import math

from spatial.spatial_processor import SpatialProcessor


def item(track_id, class_name, box, relative_depth=0.3, **extra):
    return {
        "track_id": track_id,
        "class_name": class_name,
        "confidence": 0.9,
        "bounding_box": box,
        "relative_depth": relative_depth,
        **extra,
    }


def test_processes_complete_object_and_preserves_input_fields():
    source = item(2, "bottle", [10, 20, 30, 40], 0.3, source="module5")

    result = SpatialProcessor().process([source], (100, 100, 3))

    assert result == [
        {
            **source,
            "horizontal_position": "left",
            "distance_category": "close",
        }
    ]
    assert "horizontal_position" not in source


def test_processes_multiple_objects():
    objects = [
        item(1, "person", [0, 0, 20, 20], 0.1),
        item(2, "chair", [80, 0, 100, 20], 0.9),
    ]

    result = SpatialProcessor().process(objects, (100, 100))

    assert [(obj["horizontal_position"], obj["distance_category"]) for obj in result] == [
        ("left", "very_close"),
        ("right", "far"),
    ]


def test_missing_depth_keeps_position_and_sets_category_unknown():
    result = SpatialProcessor().process([item(3, "table", [40, 10, 60, 30], None)], (100, 100))

    assert result[0]["horizontal_position"] == "center"
    assert result[0]["distance_category"] is None


def test_invalid_object_data_is_handled_without_dropping_mapping():
    source = {
        "track_id": 4,
        "class_name": "door",
        "confidence": 0.8,
        "bounding_box": [2, 2, 1, 5],
        "relative_depth": math.nan,
    }

    result = SpatialProcessor().process([source], (100, 100))

    assert result[0]["horizontal_position"] is None
    assert result[0]["distance_category"] is None
    assert result[0]["track_id"] == 4


def test_invalid_image_shape_does_not_produce_position():
    result = SpatialProcessor().process([item(5, "person", [5, 5, 15, 15])], (0, 100))

    assert result[0]["horizontal_position"] is None
    assert result[0]["distance_category"] == "close"


def test_empty_objects_returns_empty_list():
    assert SpatialProcessor().process([], (100, 100)) == []