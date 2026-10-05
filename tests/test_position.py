from spatial.position import HorizontalPositionEstimator, PositionThresholds


def test_classifies_left_center_and_right_objects():
    estimator = HorizontalPositionEstimator()
    image_shape = (100, 100, 3)

    assert estimator.classify([0, 10, 20, 30], image_shape) == "left"
    assert estimator.classify([40, 10, 60, 30], image_shape) == "center"
    assert estimator.classify([80, 10, 100, 30], image_shape) == "right"


def test_position_threshold_boundaries_classify_as_center():
    estimator = HorizontalPositionEstimator(PositionThresholds(left_boundary=0.3, right_boundary=0.7))

    assert estimator.classify([20, 10, 40, 30], (100, 100)) == "center"
    assert estimator.classify([60, 10, 80, 30], (100, 100)) == "center"


def test_clips_partial_box_at_image_boundary():
    estimator = HorizontalPositionEstimator()

    assert estimator.classify([-20, 0, 10, 20], (100, 100)) == "left"
    assert estimator.classify([90, 0, 120, 20], (100, 100)) == "right"


def test_invalid_boxes_or_image_shapes_return_none():
    estimator = HorizontalPositionEstimator()

    for box in (None, [1, 2, 3], [5, 0, 1, 4], [0, 0, float("nan"), 4], [-20, 1, -2, 5]):
        assert estimator.classify(box, (100, 100)) is None

    assert estimator.classify([1, 1, 5, 5], (0, 100)) is None