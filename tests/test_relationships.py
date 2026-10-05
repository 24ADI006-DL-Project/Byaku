from scene_graph.relationships import RelationshipGenerator, RelationshipThresholds


def obj(track_id, position="center", category="close", depth=0.3, box=None):
    return {
        "track_id": track_id,
        "class_name": "object",
        "confidence": 0.9,
        "bounding_box": box or [track_id * 100, 0, track_id * 100 + 40, 40],
        "relative_depth": depth,
        "horizontal_position": position,
        "distance_category": category,
    }


def relation_set(relationships):
    return {(rel["source"], rel["relationship"], rel["target"]) for rel in relationships}


def test_generates_left_and_right_relationships():
    relationships = RelationshipGenerator().generate(
        [obj(1, position="left"), obj(2, position="right", category="far", depth=0.9)]
    )
    edges = relation_set(relationships)

    assert (1, "left_of", 2) in edges
    assert (2, "right_of", 1) in edges


def test_generates_near_and_far_relationships_from_categories():
    relationships = RelationshipGenerator().generate(
        [
            obj(1, category="very_close", depth=0.1),
            obj(2, category="close", depth=0.3),
            obj(3, category="far", depth=0.9),
        ]
    )
    edges = relation_set(relationships)

    assert (1, "near", 2) in edges or (2, "near", 1) in edges
    assert (1, "far", 3) in edges or (3, "far", 1) in edges


def test_generates_front_and_behind_relationships():
    edges = relation_set(
        RelationshipGenerator().generate(
            [obj(1, depth=0.1), obj(2, depth=0.8)]
        )
    )

    assert (1, "in_front_of", 2) in edges
    assert (2, "behind", 1) in edges


def test_generates_next_to_only_within_pixel_threshold():
    generator = RelationshipGenerator(RelationshipThresholds(next_to_pixel_gap=10))
    close = generator.generate(
        [obj(1, box=[0, 0, 20, 20]), obj(2, box=[25, 0, 45, 20])]
    )
    distant = generator.generate(
        [obj(1, box=[0, 0, 20, 20]), obj(2, box=[40, 0, 60, 20])]
    )

    assert any(rel["relationship"] == "next_to" for rel in close)
    assert not any(rel["relationship"] == "next_to" for rel in distant)


def test_relationship_thresholds_can_suppress_edges():
    generator = RelationshipGenerator(
        RelationshipThresholds(front_depth_delta=0.5, near_max_category_rank=0, far_min_category_gap=3)
    )
    relationships = generator.generate(
        [obj(1, position="left", category="close", depth=0.2), obj(2, position="right", category="moderate", depth=0.4)]
    )
    edges = relation_set(relationships)

    assert (1, "left_of", 2) in edges
    assert not any(rel in {"near", "far", "in_front_of", "behind"} for _, rel, _ in edges)


def test_does_not_create_self_relationships():
    relationships = RelationshipGenerator().generate([obj(9, position="left", depth=0.1)])

    assert all(rel["source"] != rel["target"] for rel in relationships)