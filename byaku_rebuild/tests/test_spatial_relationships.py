import numpy as np
import pytest

from byaku_rebuild.spatial_relationships import (
    bounding_box_iou,
    calculate_spatial_relationships,
    image_shape_from_frame_size,
)


def make_detection(
    box: list[float],
    *,
    class_name: str = "object",
    track_id: int | None = None,
    box_key: str = "bounding_box",
    confidence: float | None = None,
) -> dict:
    detection = {"class_name": class_name, box_key: box}
    if track_id is not None:
        detection["track_id"] = track_id
    if confidence is not None:
        detection["confidence"] = confidence
    return detection


def get_relation(result: dict, relation: str) -> dict:
    return next(item for item in result["relationships"] if item["relation"] == relation)


def test_left_right_and_above_below_use_box_centers() -> None:
    result = calculate_spatial_relationships(
        [
            make_detection([5, 5, 15, 15], track_id=1),
            make_detection([75, 75, 85, 85], track_id=2),
        ],
        (100, 100),
        center_zone_fraction=0.1,
    )

    assert result["objects"][0]["horizontal_position"] == "left"
    assert result["objects"][0]["vertical_position"] == "above"
    assert result["objects"][1]["horizontal_position"] == "right"
    assert result["objects"][1]["vertical_position"] == "below"
    assert get_relation(result, "left_of")["subject"] == "track:1"
    assert get_relation(result, "above")["subject"] == "track:1"
    assert get_relation(result, "right_of")["subject"] == "track:2"
    assert get_relation(result, "below")["subject"] == "track:2"


def test_center_zone_classification_includes_boundaries() -> None:
    result = calculate_spatial_relationships(
        [
            make_detection([30, 30, 40, 40]),
            make_detection([60, 60, 70, 70]),
        ],
        (100, 100),
        center_zone_fraction=0.2,
    )

    assert [item["horizontal_position"] for item in result["objects"]] == ["center", "center"]
    assert [item["vertical_position"] for item in result["objects"]] == ["middle", "middle"]


def test_bounding_box_overlap_uses_configurable_iou_threshold() -> None:
    first = make_detection([10, 10, 30, 30], track_id=4)
    second = make_detection([20, 20, 40, 40], track_id=5)

    assert bounding_box_iou(first["bounding_box"], second["bounding_box"]) == pytest.approx(1 / 7)
    result = calculate_spatial_relationships([first, second], (50, 50), overlap_threshold=0.14)
    assert get_relation(result, "overlapping")["iou"] == pytest.approx(1 / 7)
    assert not any(
        item["relation"] == "overlapping"
        for item in calculate_spatial_relationships([first, second], (50, 50), overlap_threshold=0.15)["relationships"]
    )


def test_relative_depth_uses_median_pixels_inside_each_box() -> None:
    depth = np.zeros((20, 40), dtype=np.float32)
    depth[:, :20] = 0.8
    depth[:, 20:] = 0.2
    result = calculate_spatial_relationships(
        [
            make_detection([0, 0, 10, 10], track_id=1),
            make_detection([30, 0, 40, 10], track_id=2),
        ],
        (20, 40),
        depth,
    )

    assert [item["relative_depth_value"] for item in result["objects"]] == pytest.approx([0.8, 0.2])
    relation = get_relation(result, "nearer_than")
    assert (relation["subject"], relation["object"]) == ("track:1", "track:2")
    assert relation["depth_difference"] == pytest.approx(0.6)


def test_depth_direction_is_configurable_and_similar_values_use_tolerance() -> None:
    depth = np.zeros((10, 20), dtype=np.float32)
    depth[:, :10] = 0.8
    depth[:, 10:] = 0.2
    detections = [make_detection([0, 0, 10, 10]), make_detection([10, 0, 20, 10])]

    inverse_depth = calculate_spatial_relationships(detections, (10, 20), depth)
    metric_like = calculate_spatial_relationships(detections, (10, 20), depth, higher_depth_is_nearer=False)
    similar = calculate_spatial_relationships(
        detections,
        (10, 20),
        np.where(np.indices((10, 20))[1] < 10, 0.8, 0.77).astype(np.float32),
        depth_difference_threshold=0.05,
    )

    assert get_relation(inverse_depth, "nearer_than")["subject"] == inverse_depth["objects"][0]["object_id"]
    assert get_relation(metric_like, "farther_than")["subject"] == metric_like["objects"][0]["object_id"]
    assert get_relation(similar, "similar_depth")["depth_difference"] == pytest.approx(0.03, abs=1e-6)


def test_missing_depth_keeps_geometric_relations_only() -> None:
    result = calculate_spatial_relationships(
        [make_detection([0, 0, 10, 10]), make_detection([20, 20, 30, 30])],
        (40, 40),
    )

    assert all(item["relative_depth_value"] is None for item in result["objects"])
    assert {item["relation"] for item in result["relationships"]} == {"left_of", "right_of", "above", "below"}
    assert result["warnings"] == []


def test_invalid_and_non_finite_depth_maps_are_ignored_with_warning() -> None:
    detections = [make_detection([0, 0, 10, 10]), make_detection([20, 20, 30, 30])]
    for invalid_depth in (np.zeros((4, 4, 3)), np.full((40, 40), np.nan, dtype=np.float32)):
        result = calculate_spatial_relationships(detections, (40, 40), invalid_depth)
        assert all(item["relative_depth_value"] is None for item in result["objects"])
        assert not any("depth" in item["relation"] or item["relation"] in {"nearer_than", "farther_than"}
                       for item in result["relationships"])
        assert result["warnings"]


def test_mismatched_depth_dimensions_skip_depth_but_keep_2d_relations() -> None:
    result = calculate_spatial_relationships(
        [make_detection([0, 0, 10, 10]), make_detection([20, 20, 30, 30])],
        (40, 40),
        np.ones((20, 20), dtype=np.float32),
    )

    assert {item["relation"] for item in result["relationships"]} == {"left_of", "right_of", "above", "below"}
    assert all(item["relative_depth_value"] is None for item in result["objects"])
    assert "do not match" in result["warnings"][0]


def test_invalid_boxes_are_skipped_and_partially_outside_boxes_are_clipped() -> None:
    result = calculate_spatial_relationships(
        [
            make_detection([5, 5, 5, 10]),
            make_detection([float("nan"), 0, 10, 10]),
            make_detection([100, 100, 120, 120]),
            make_detection([-5, -3, 12, 14], track_id=9),
        ],
        (20, 20),
    )

    assert len(result["objects"]) == 1
    assert result["objects"][0]["bounding_box"] == (0.0, 0.0, 12.0, 14.0)
    assert len(result["warnings"]) == 3


def test_empty_detections_and_invalid_image_shape_are_safe() -> None:
    empty = calculate_spatial_relationships([], (10, 10), np.ones((10, 10)))
    invalid_shape = calculate_spatial_relationships([make_detection([0, 0, 1, 1])], (0, 10))

    assert empty["objects"] == []
    assert empty["relationships"] == []
    assert invalid_shape["objects"] == []
    assert invalid_shape["warnings"]


def test_missing_tracking_ids_receive_canonical_frame_specific_ids() -> None:
    result = calculate_spatial_relationships(
        [make_detection([0, 0, 10, 10], box_key="bbox"), make_detection([20, 20, 30, 30])],
        (40, 40),
        frame_id=42,
    )

    assert [item["object_id"] for item in result["objects"]] == ["temporary:42:0", "temporary:42:1"]
    assert all(item["track_id"] is None for item in result["objects"])


def test_untracked_ids_differ_between_automatically_identified_frames() -> None:
    detections = [make_detection([0, 0, 10, 10])]

    first = calculate_spatial_relationships(detections, (20, 40))
    second = calculate_spatial_relationships(detections, (20, 40))

    assert first["frame_id"] != second["frame_id"]
    assert first["objects"][0]["object_id"] != second["objects"][0]["object_id"]
    assert first["objects"][0]["object_id"].startswith("temporary:")


def test_confidence_is_preserved_when_supplied_and_optional_when_absent() -> None:
    result = calculate_spatial_relationships(
        [make_detection([0, 0, 10, 10], confidence=0.87), make_detection([20, 0, 30, 10])],
        (20, 40),
        frame_id=1,
    )

    assert result["objects"][0]["confidence"] == pytest.approx(0.87)
    assert "confidence" not in result["objects"][1]


def test_api_frame_size_conversion_preserves_non_square_dimensions() -> None:
    image_shape = image_shape_from_frame_size({"width": 80, "height": 40})
    result = calculate_spatial_relationships(
        [make_detection([2, 15, 12, 25])],
        image_shape,
        frame_id=1,
    )

    assert image_shape == (40, 80)
    assert result["image_shape"] == (40, 80)
    assert result["objects"][0]["horizontal_position"] == "left"
    assert result["objects"][0]["vertical_position"] == "middle"


def test_external_frame_size_is_explicitly_converted_without_guessing() -> None:
    assert image_shape_from_frame_size({"width": 40, "height": 80}) == (80, 40)
    with pytest.raises(ValueError):
        image_shape_from_frame_size({"width": 40})


def test_depth_roi_clips_at_image_edges_and_ignores_non_finite_pixels() -> None:
    depth = np.full((10, 10), np.nan, dtype=np.float32)
    depth[0:5, 0:5] = 0.7
    result = calculate_spatial_relationships(
        [make_detection([-5, -5, 5, 5], track_id=2)],
        (10, 10),
        depth,
    )

    assert result["objects"][0]["bounding_box"] == (0.0, 0.0, 5.0, 5.0)
    assert result["objects"][0]["relative_depth_value"] == pytest.approx(0.7)


@pytest.mark.parametrize(
    ("option", "value"),
    [
        ("center_zone_fraction", 0.5),
        ("overlap_threshold", 0),
        ("depth_difference_threshold", -0.1),
    ],
)
def test_invalid_thresholds_raise_value_error(option: str, value: float) -> None:
    with pytest.raises(ValueError):
        calculate_spatial_relationships([], (10, 10), **{option: value})