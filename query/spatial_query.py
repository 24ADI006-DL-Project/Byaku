from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Literal, TypedDict

from query.query_processor import ParsedQuery, QueryProcessor


QueryResultStatus = Literal[
    "found",
    "not_found",
    "unsupported",
    "ambiguous",
    "unknown_object",
    "empty_query",
]


class SpatialQueryResult(TypedDict):
    intent: str | None
    found: bool
    target: str | None
    objects: list[dict[str, Any]]
    status: QueryResultStatus


class SpatialQueryEngine:
    """Retrieve matching objects and stored spatial attributes from a scene."""

    def __init__(self, scene: Any, query_processor: QueryProcessor | None = None) -> None:
        self.scene = scene
        self.query_processor = query_processor or QueryProcessor()

    def answer(self, query: str) -> SpatialQueryResult:
        parsed = self.query_processor.parse(query)
        intent = parsed["intent"]
        target = parsed["target"]
        parse_status = parsed["status"]

        if parse_status == "empty_query":
            return self._result(intent, target, [], "empty_query")
        if parse_status == "unsupported":
            return self._result(intent, target, [], "unsupported")
        if parse_status == "ambiguous":
            return self._result(intent, target, [], "ambiguous")
        if target is None:
            return self._result(intent, target, [], "unknown_object")

        canonical_target = self.query_processor.canonicalize_class_name(target)
        matches = [
            obj
            for obj in self._current_objects()
            if self.query_processor.canonicalize_class_name(obj.get("class_name", ""))
            == canonical_target
        ]
        if not matches:
            return self._result(intent, target, [], "not_found")
        return self._result(intent, target, matches, "found")

    def _current_objects(self) -> list[dict[str, Any]]:
        memory_objects = getattr(self.scene, "objects", None)
        if memory_objects is not None:
            if callable(memory_objects):
                memory_objects = memory_objects()
            if isinstance(memory_objects, (list, tuple)):
                return [dict(obj) for obj in memory_objects if isinstance(obj, Mapping)]

        graph_owner = getattr(self.scene, "scene_graph", self.scene)
        graph = getattr(graph_owner, "graph", graph_owner)
        nodes = getattr(graph, "nodes", None)
        if nodes is None:
            return []

        try:
            entries = nodes(data=True) if callable(nodes) else nodes.items()
        except (AttributeError, TypeError):
            return []

        objects: list[dict[str, Any]] = []
        for entry in entries:
            if not isinstance(entry, tuple) or len(entry) != 2:
                continue
            track_id, attributes = entry
            if not isinstance(attributes, Mapping):
                continue
            obj = dict(attributes)
            obj.setdefault("track_id", track_id)
            objects.append(obj)
        return objects

    @staticmethod
    def _result(
        intent: str | None,
        target: str | None,
        objects: list[dict[str, Any]],
        status: QueryResultStatus,
    ) -> SpatialQueryResult:
        return {
            "intent": intent,
            "found": bool(objects),
            "target": target,
            "objects": objects,
            "status": status,
        }