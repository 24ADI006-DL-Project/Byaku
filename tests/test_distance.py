import math

from spatial.distance import RelativeDepthEstimator, RelativeDepthThresholds


def test_relative_depth_maps_to_each_qualitative_category():
    estimator = RelativeDepthEstimator()

    assert estimator.estimate_category(0.1) == "very_close"
    assert estimator.estimate_category(0.3) == "close"
    assert estimator.estimate_category(0.6) == "moderate"
    assert estimator.estimate_category(0.9) == "far"


def test_threshold_boundaries_belong_to_nearer_category():
    estimator = RelativeDepthEstimator()

    assert estimator.estimate_category(0.25) == "very_close"
    assert estimator.estimate_category(0.5) == "close"
    assert estimator.estimate_category(0.75) == "moderate"


def test_distance_thresholds_are_configurable():
    estimator = RelativeDepthEstimator(
        RelativeDepthThresholds(very_close_max=0.1, close_max=0.3, moderate_max=0.6)
    )

    assert estimator.estimate_category(0.2) == "close"
    assert estimator.estimate_category(0.5) == "moderate"


def test_invalid_relative_depth_has_no_category():
    estimator = RelativeDepthEstimator()

    for value in (None, "invalid", math.nan, math.inf, -0.1):
        assert estimator.estimate_category(value) is None