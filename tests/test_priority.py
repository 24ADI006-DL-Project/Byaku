from reasoning.priority import PriorityConfig, PriorityScorer


def spatial_object(class_name="bottle", distance_category="moderate"):
    return {"class_name": class_name, "distance_category": distance_category}


def test_priority_calculation_returns_score_tier_and_decision():
    result = PriorityScorer().assess(spatial_object())

    assert result == {"score": 1.0, "priority": "low", "should_announce": False}


def test_very_close_object_gets_higher_priority():
    result = PriorityScorer().assess(spatial_object(distance_category="very_close"))

    assert result["score"] == 4.0
    assert result["priority"] == "medium"
    assert result["should_announce"] is True


def test_far_object_gets_low_priority():
    result = PriorityScorer().assess(spatial_object(distance_category="far"))

    assert result["priority"] == "low"
    assert result["should_announce"] is False


def test_important_class_increases_priority():
    result = PriorityScorer().assess(spatial_object("person", "far"))

    assert result["score"] == 3.0
    assert result["priority"] == "medium"
    assert result["should_announce"] is True


def test_new_moving_and_position_changing_object_gains_priority():
    result = PriorityScorer().assess(
        spatial_object("chair", "far"),
        newly_detected=True,
        significant_movement=True,
        significant_position_change=True,
    )

    assert result["score"] == 5.0
    assert result["priority"] == "medium"
    assert result["should_announce"] is True


def test_recent_announcement_reduces_score_and_suppresses_announcement():
    scorer = PriorityScorer(PriorityConfig(announcement_threshold=5))
    result = scorer.assess(
        spatial_object("person", "very_close"),
        recently_announced=True,
    )

    assert result["score"] == 4.0
    assert result["should_announce"] is False


def test_priority_thresholds_and_weights_are_configurable():
    scorer = PriorityScorer(
        PriorityConfig(
            distance_points={"very_close": 0, "close": 0, "moderate": 0, "far": 0},
            important_classes=frozenset({"bottle"}),
            important_class_bonus=8,
            high_threshold=8,
            medium_threshold=4,
            announcement_threshold=8,
        )
    )

    result = scorer.assess(spatial_object("bottle", "far"))

    assert result["score"] == 8.0
    assert result["priority"] == "high"
    assert result["should_announce"] is True


def test_invalid_distance_category_does_not_fail():
    result = PriorityScorer().assess(spatial_object(distance_category="unknown"))

    assert result["score"] == 0.0
    assert result["priority"] == "low"