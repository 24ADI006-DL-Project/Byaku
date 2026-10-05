from narration.narrator import Narrator


def query_result(class_name="bottle", position="left", category="close", **result_fields):
    obj = {"class_name": class_name}
    if position is not None:
        obj["horizontal_position"] = position
    if category is not None:
        obj["distance_category"] = category
    return {"intent": "locate", "found": True, "target": class_name, "objects": [obj], **result_fields}


def test_narrates_a_found_object():
    assert Narrator().narrate(query_result()) == "Your bottle is close and on your left."


def test_narrates_an_object_that_is_not_found():
    result = {"intent": "locate", "found": False, "target": "bottle", "objects": []}

    assert Narrator().narrate(result) == "The bottle is not currently visible."


def test_narrates_left_center_and_right_positions():
    narrator = Narrator()

    assert narrator.narrate(query_result(position="left", category=None)) == "Your bottle is on your left."
    assert narrator.narrate(query_result(position="center", category=None)) == "Your bottle is in front of you."
    assert narrator.narrate(query_result(position="right", category=None)) == "Your bottle is on your right."


def test_narrates_each_distance_category():
    narrator = Narrator()

    assert narrator.narrate(query_result(position=None, category="very_close")) == "Your bottle is very close."
    assert narrator.narrate(query_result(position=None, category="close")) == "Your bottle is close."
    assert narrator.narrate(query_result(position=None, category="moderate")) == "Your bottle is moderately far."
    assert narrator.narrate(query_result(position=None, category="far")) == "Your bottle is far."


def test_narrates_multiple_objects_without_track_ids():
    result = {
        "intent": "locate",
        "found": True,
        "target": "bottle",
        "objects": [
            {"track_id": 7, "class_name": "bottle", "horizontal_position": "left", "distance_category": "close"},
            {"track_id": 9, "class_name": "bottle", "horizontal_position": "right", "distance_category": "far"},
        ],
    }

    text = Narrator().narrate(result)

    assert text == "There are two bottles. One is close on your left and one is farther on your right."
    assert "7" not in text
    assert "9" not in text


def test_missing_position_and_distance_are_omitted_without_guessing():
    narrator = Narrator()

    assert narrator.narrate(query_result(position=None, category="close")) == "Your bottle is close."
    assert narrator.narrate(query_result(position="right", category=None)) == "Your bottle is on your right."


def test_existence_intent_uses_a_check_template():
    result = query_result(class_name="chair", position="center", category="moderate", intent="existence")

    assert Narrator().narrate(result) == "Yes, there is a chair moderately far and in front of you."


def test_empty_or_invalid_result_is_handled():
    narrator = Narrator()

    assert narrator.narrate({}) == "I can't determine the scene information right now."
    assert narrator.narrate(None) == "I can't determine the scene information right now."
    assert narrator.narrate({"found": True, "objects": []}) == "I can't determine the object's location from the available information."