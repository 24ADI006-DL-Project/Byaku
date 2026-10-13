from collections.abc import Mapping
from itertools import count
from typing import Any

import networkx as nx


RELATION_TYPES = {
    "left_of",
    "right_of",
    "above",
    "below",
    "nearer_than",
    "farther_than",
    "similar_depth",
    "overlapping",
}
_GRAPH_FRAME_COUNTER = count(1)


def _warning(graph: nx.MultiDiGraph, message: str) -> None:
    graph.graph.setdefault("warnings", []).append(message)


def _edge_record(
    source: Any,
    destination: Any,
    key: Any,
    attributes: Mapping[str, Any],
    direction: str,
) -> dict[str, Any]:
    return {
        "subject": source,
        "object": destination,
        "relation": attributes.get("relation", key),
        "direction": direction,
        **dict(attributes),
    }


def build_scene_graph(spatial_output: Mapping[str, Any] | None) -> nx.MultiDiGraph:
    """Build a directed relationship graph from calculate_spatial_relationships output."""
    graph = nx.MultiDiGraph()
    graph.graph["warnings"] = []
    if not isinstance(spatial_output, Mapping):
        _warning(graph, "spatial output must be a mapping; built an empty graph")
        return graph

    graph.graph["image_shape"] = spatial_output.get("image_shape")
    frame_id = spatial_output.get("frame_id")
    if isinstance(frame_id, bool) or not isinstance(frame_id, (str, int)) or not str(frame_id).strip():
        frame_id = f"graph-auto-{next(_GRAPH_FRAME_COUNTER)}"
    else:
        frame_id = str(frame_id)
    graph.graph["frame_id"] = frame_id
    id_aliases: dict[Any, Any] = {}
    upstream_warnings = spatial_output.get("warnings", [])
    if isinstance(upstream_warnings, (list, tuple)):
        graph.graph["warnings"].extend(str(message) for message in upstream_warnings)

    objects = spatial_output.get("objects", [])
    if not isinstance(objects, (list, tuple)):
        _warning(graph, "objects must be a list or tuple; no nodes were added")
        objects = []

    for index, item in enumerate(objects):
        if not isinstance(item, Mapping):
            _warning(graph, f"object {index} is not a mapping and was skipped")
            continue

        original_id = item.get("object_id")
        track_id = item.get("track_id")
        if isinstance(original_id, str) and original_id.startswith("temporary:") and track_id is None:
            object_id = original_id
        elif isinstance(original_id, str) and original_id.startswith("frame:"):
            if track_id is not None:
                object_id = track_id if isinstance(track_id, str) and track_id.startswith("track:") else f"track:{track_id}"
            else:
                suffix = original_id[len("frame:"):] or str(index)
                object_id = f"temporary:{frame_id}:{suffix}"
        elif original_id is not None:
            object_id = original_id
        elif track_id is not None:
            object_id = track_id if isinstance(track_id, str) and track_id.startswith("track:") else f"track:{track_id}"
        else:
            object_id = f"temporary:{frame_id}:{index}"
        try:
            hash(object_id)
        except TypeError:
            _warning(graph, f"object {index} has an unhashable ID and was skipped")
            continue

        if graph.has_node(object_id):
            _warning(graph, f"duplicate object ID {object_id!r}; retained the first node")
            continue

        attributes = dict(item)
        attributes["object_id"] = object_id
        graph.add_node(object_id, **attributes)
        if original_id is not None:
            id_aliases[original_id] = object_id

    relationships = spatial_output.get("relationships", [])
    if not isinstance(relationships, (list, tuple)):
        _warning(graph, "relationships must be a list or tuple; no edges were added")
        return graph

    for index, item in enumerate(relationships):
        if not isinstance(item, Mapping):
            _warning(graph, f"relationship {index} is not a mapping and was skipped")
            continue

        source = item.get("subject")
        destination = item.get("object")
        try:
            source = id_aliases.get(source, source)
            destination = id_aliases.get(destination, destination)
        except TypeError:
            pass
        relation_type = item.get("relation")
        if source is None or destination is None or not isinstance(relation_type, str):
            _warning(graph, f"relationship {index} is missing subject, object, or relation and was skipped")
            continue
        if relation_type not in RELATION_TYPES:
            _warning(graph, f"relationship {index} has unsupported type {relation_type!r} and was skipped")
            continue
        try:
            hash(source)
            hash(destination)
        except TypeError:
            _warning(graph, f"relationship {index} references an unhashable object ID and was skipped")
            continue
        if source not in graph or destination not in graph or source == destination:
            _warning(graph, f"relationship {index} references a missing or identical object and was skipped")
            continue

        if graph.has_edge(source, destination, key=relation_type):
            _warning(
                graph,
                f"duplicate {relation_type!r} relationship from {source!r} to {destination!r}; skipped",
            )
            continue

        evidence = {
            name: value
            for name, value in item.items()
            if name not in {"subject", "object", "relation"}
        }
        graph.add_edge(source, destination, key=relation_type, relation=relation_type, **evidence)

    return graph


def get_objects(graph: nx.MultiDiGraph | None) -> list[dict[str, Any]]:
    """Return graph nodes as object records, or an empty list for invalid input."""
    if not isinstance(graph, nx.MultiDiGraph):
        return []
    return [{"object_id": object_id, **dict(attributes)} for object_id, attributes in graph.nodes(data=True)]


def find_relationships_for_object(
    graph: nx.MultiDiGraph | None,
    object_id: Any,
) -> list[dict[str, Any]]:
    """Return outgoing and incoming edges for one object, preserving edge direction."""
    if not isinstance(graph, nx.MultiDiGraph) or object_id not in graph:
        return []

    relationships = [
        _edge_record(source, destination, key, attributes, "outgoing")
        for source, destination, key, attributes in graph.out_edges(object_id, keys=True, data=True)
    ]
    relationships.extend(
        _edge_record(source, destination, key, attributes, "incoming")
        for source, destination, key, attributes in graph.in_edges(object_id, keys=True, data=True)
    )
    return relationships


def find_objects_by_class(
    graph: nx.MultiDiGraph | None,
    class_name: str,
) -> list[dict[str, Any]]:
    """Return objects whose class matches case-insensitively."""
    if not isinstance(graph, nx.MultiDiGraph):
        return []
    target_class = str(class_name).casefold()
    return [
        {"object_id": object_id, **dict(attributes)}
        for object_id, attributes in graph.nodes(data=True)
        if str(attributes.get("class_name", "")).casefold() == target_class
    ]


def query_relationship(
    graph: nx.MultiDiGraph | None,
    source: Any,
    destination: Any,
    relationship_type: str | None = None,
) -> list[dict[str, Any]]:
    """Return directed relationships from source to destination, optionally by type."""
    if not isinstance(graph, nx.MultiDiGraph) or source not in graph or destination not in graph:
        return []

    edges = graph.get_edge_data(source, destination, default={})
    records = []
    for key, attributes in edges.items():
        relation_type = attributes.get("relation", key)
        if relationship_type is None or relation_type == relationship_type:
            records.append(_edge_record(source, destination, key, attributes, "outgoing"))
    return records