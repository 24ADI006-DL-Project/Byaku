from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from scene_graph.scene_graph import SceneGraph


class SceneMemory:
    """Short-term store of the latest object state, bounded by missed frames."""

    def __init__(self, max_missing_frames: int = 30, scene_graph: SceneGraph | None = None) -> None:
        if max_missing_frames < 1:
            raise ValueError("max_missing_frames must be at least 1")
        self.max_missing_frames = int(max_missing_frames)
        self.scene_graph = scene_graph or SceneGraph()
        self._frame_number = 0
        self._objects: dict[Any, dict[str, Any]] = {}

    def update(self, spatial_objects: Sequence[Mapping[str, Any]] | None) -> list[dict[str, Any]]:
        """Observe one processed frame, update tracks, and expire stale objects."""
        self._frame_number += 1
        observed: dict[Any, dict[str, Any]] = {}

        for spatial_object in spatial_objects or []:
            if not isinstance(spatial_object, Mapping) or "track_id" not in spatial_object:
                continue
            track_id = spatial_object["track_id"]
            try:
                hash(track_id)
            except TypeError:
                continue
            observed[track_id] = dict(spatial_object)

        for track_id, spatial_object in observed.items():
            self._objects[track_id] = {
                "spatial_object": spatial_object,
                "last_seen_frame": self._frame_number,
            }

        expired = [
            track_id
            for track_id, entry in self._objects.items()
            if track_id not in observed
            and self._frame_number - entry["last_seen_frame"] >= self.max_missing_frames
        ]
        for track_id in expired:
            del self._objects[track_id]

        self.scene_graph.update_objects(list(observed.values()))
        if expired:
            self.scene_graph.remove_objects(expired)
        else:
            self.scene_graph.refresh_relationships()

        return self.objects

    @property
    def objects(self) -> list[dict[str, Any]]:
        """Return copies of the current short-term memory entries."""
        return [dict(entry["spatial_object"]) for entry in self._objects.values()]

    @property
    def frame_number(self) -> int:
        return self._frame_number

    def reset(self) -> None:
        """Clear stored tracks and the associated scene graph."""
        self._objects.clear()
        self._frame_number = 0
        self.scene_graph.graph.clear()