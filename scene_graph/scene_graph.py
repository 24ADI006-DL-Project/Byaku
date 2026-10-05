from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import networkx as nx

from scene_graph.relationships import RelationshipGenerator


class SceneGraph:
    """NetworkX representation of the latest known spatial objects."""

    def __init__(self, relationship_generator: RelationshipGenerator | None = None) -> None:
        self.graph = nx.MultiDiGraph()
        self.relationship_generator = relationship_generator or RelationshipGenerator()

    def add_object(self, spatial_object: Mapping[str, Any]) -> Any:
        """Add a node, replacing its attributes if the track ID already exists."""
        track_id = self._track_id(spatial_object)
        if track_id in self.graph:
            self.graph.nodes[track_id].clear()
        self.graph.add_node(track_id, **dict(spatial_object))
        self.refresh_relationships()
        return track_id

    def update_object(self, spatial_object: Mapping[str, Any]) -> Any:
        """Store the latest spatial data for an object's track ID."""
        return self.add_object(spatial_object)

    def update_objects(self, spatial_objects: Sequence[Mapping[str, Any]]) -> None:
        """Add or update a batch of objects, retaining unseen nodes."""
        for spatial_object in spatial_objects:
            track_id = self._track_id(spatial_object)
            if track_id in self.graph:
                self.graph.nodes[track_id].clear()
            self.graph.add_node(track_id, **dict(spatial_object))
        self.refresh_relationships()

    def remove_object(self, track_id: Any) -> bool:
        """Remove a node by ID and refresh relationships; return whether it existed."""
        if track_id not in self.graph:
            return False
        self.graph.remove_node(track_id)
        self.refresh_relationships()
        return True

    def remove_objects(self, track_ids: Sequence[Any]) -> None:
        """Remove multiple nodes and refresh relationships once."""
        self.graph.remove_nodes_from(track_ids)
        self.refresh_relationships()

    def refresh_relationships(self) -> None:
        """Rebuild edges from current node attributes using deterministic rules."""
        self.graph.remove_edges_from(list(self.graph.edges(keys=True)))
        objects = [dict(attributes) for _, attributes in self.graph.nodes(data=True)]
        for relationship in self.relationship_generator.generate(objects):
            source = relationship["source"]
            target = relationship["target"]
            if source in self.graph and target in self.graph and source != target:
                relation = relationship["relationship"]
                self.graph.add_edge(source, target, key=relation, relationship=relation)

    @property
    def relationships(self) -> list[dict[str, Any]]:
        return [
            {"source": source, "relationship": data["relationship"], "target": target}
            for source, target, _, data in self.graph.edges(keys=True, data=True)
        ]

    @staticmethod
    def _track_id(spatial_object: Mapping[str, Any]) -> Any:
        if "track_id" not in spatial_object:
            raise ValueError("spatial object must contain track_id")
        track_id = spatial_object["track_id"]
        try:
            hash(track_id)
        except TypeError as exc:
            raise ValueError("track_id must be hashable") from exc
        return track_id