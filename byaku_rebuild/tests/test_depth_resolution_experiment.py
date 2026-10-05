import base64

import cv2
import numpy as np
import pytest

from byaku_rebuild.experiment_depth_resolution import (
    INPUT_POLICIES,
    object_depth_category_agreement,
    _preview_check,
    _statistics,
    pairwise_depth_order_agreement,
)


def test_experiment_keeps_baseline_and_required_input_policies():
    assert INPUT_POLICIES == (
        ("baseline_364", 364),
        ("shorter_side_320", 320),
        ("shorter_side_256", 256),
        ("shorter_side_224", 224),
    )


def test_statistics_report_all_required_summary_values():
    assert _statistics([1, 2, 3, 4]) == {
        "count": 4,
        "mean_ms": 2.5,
        "median_ms": 2.5,
        "min_ms": 1.0,
        "max_ms": 4.0,
    }


def test_pairwise_depth_agreement_counts_only_comparable_orders():
    baseline = [
        {
            "objects": [
                {"object_id": "a", "relative_depth_value": 0.9},
                {"object_id": "b", "relative_depth_value": 0.2},
                {"object_id": "c", "relative_depth_value": 0.51},
            ]
        }
    ]
    candidate = [
        {
            "objects": [
                {"object_id": "a", "relative_depth_value": 0.75},
                {"object_id": "b", "relative_depth_value": 0.1},
                {"object_id": "c", "relative_depth_value": 0.8},
            ]
        }
    ]

    result = pairwise_depth_order_agreement(baseline, candidate)

    assert result["comparable_pairs"] == 3
    assert result["preserved_orders"] == 2
    assert result["agreement"] == pytest.approx(2 / 3)
    assert len(result["disagreements"]) == 1


def test_pairwise_depth_agreement_reports_no_valid_comparisons():
    frames = [{"objects": [{"object_id": "a", "relative_depth_value": None}]}]

    result = pairwise_depth_order_agreement(frames, frames)

    assert result["comparable_pairs"] == 0
    assert result["agreement"] is None
    assert result["agreement_percent"] is None


def test_object_depth_categories_compare_relative_near_middle_far_groups():
    baseline = [{"objects": [
        {"object_id": "a", "relative_depth_value": 0.1},
        {"object_id": "b", "relative_depth_value": 0.5},
        {"object_id": "c", "relative_depth_value": 0.9},
    ]}]
    candidate = [{"objects": [
        {"object_id": "a", "relative_depth_value": 0.2},
        {"object_id": "b", "relative_depth_value": 0.4},
        {"object_id": "c", "relative_depth_value": 0.8},
    ]}]

    result = object_depth_category_agreement(baseline, candidate)

    assert result["comparable_objects"] == 3
    assert result["preserved_categories"] == 3
    assert result["agreement_percent"] == 100


def test_preview_check_decodes_valid_depth_jpeg():
    image = np.zeros((320, 180, 3), dtype=np.uint8)
    encoded, jpeg = cv2.imencode(".jpg", image)
    assert encoded

    result = _preview_check(
        {"depth_preview": base64.b64encode(jpeg.tobytes()).decode("ascii")},
        width=720,
        height=1280,
    )

    assert result == {"valid": True, "shape": [320, 180], "reason": None}


def test_preview_check_rejects_missing_payload():
    result = _preview_check({}, width=720, height=1280)

    assert result == {"valid": False, "reason": "depth_preview missing"}
