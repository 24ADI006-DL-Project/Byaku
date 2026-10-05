from query.query_processor import QueryProcessor


def test_parses_locate_query():
    parsed = QueryProcessor().parse("Where is my bottle?")

    assert parsed["intent"] == "locate"
    assert parsed["target"] == "bottle"
    assert parsed["status"] == "ok"


def test_parses_existence_query():
    parsed = QueryProcessor().parse("Is there a chair?")

    assert parsed["intent"] == "existence"
    assert parsed["target"] == "chair"


def test_extracts_configured_vocabulary_and_plural_form():
    processor = QueryProcessor(object_vocabulary=["mug", "coffee table"])

    parsed = processor.parse("Where are the mugs?")
    phrase = processor.parse("Is there a coffee table?")

    assert parsed["target"] == "mug"
    assert phrase["target"] == "coffee table"


def test_unsupported_intent_is_reported():
    parsed = QueryProcessor().parse("What color is the bottle?")

    assert parsed["intent"] is None
    assert parsed["status"] == "unsupported"


def test_unknown_object_is_not_added_to_vocabulary():
    parsed = QueryProcessor().parse("Where is my toaster?")

    assert parsed["intent"] == "locate"
    assert parsed["target"] == "toaster"
    assert parsed["status"] == "unknown_object"


def test_multiple_known_objects_are_reported_as_ambiguous():
    parsed = QueryProcessor().parse("Where is the bottle near the chair?")

    assert parsed["target"] is None
    assert parsed["status"] == "ambiguous"


def test_empty_query_is_reported():
    parsed = QueryProcessor().parse("  ")

    assert parsed["intent"] is None
    assert parsed["target"] is None
    assert parsed["status"] == "empty_query"