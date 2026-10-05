from collections.abc import Mapping, Sequence
from copy import deepcopy
from numbers import Integral
import time
from typing import Any

import networkx as nx

from .query_reasoning import answer_query
from .scene_description import describe_scene
from .scene_graph import build_scene_graph, get_objects
from .scene_memory import SceneMemory
from .spatial_relationships import calculate_spatial_relationships, image_shape_from_frame_size
from .speech_response import prepare_spoken_response, prepare_spoken_scene_summary
from .frame_profiler import FrameProfiler, profile_stage


class PipelineOrchestrator:
    """Coordinate the already-tested BYAKU modules using supplied frame data only.

    This class performs no capture, detection, tracking, depth inference, API
    handling, or speech playback. Callers supply detections/tracks and optional
    relative depth for each frame.
    """

    def __init__(self, scene_memory: SceneMemory | None = None) -> None:
        self.scene_memory = scene_memory or SceneMemory()
        self._latest_spatial: dict[str, Any] | None = None
        self._latest_graph: nx.MultiDiGraph | None = None
        self._latest_scene_description: dict[str, Any] | None = None
        self._latest_frame_id: str | None = None
        self._latest_frame_number: int | None = None
        self._latest_commit_wall_time: float | None = None
        self._seen_frame_ids: set[str] = set()

    @staticmethod
    def _dimensions(value: Any) -> tuple[int, int]:
        if isinstance(value, Mapping):
            height, width = image_shape_from_frame_size(value)
        else:
            if isinstance(value, (str, bytes)) or not isinstance(value, Sequence) or len(value) != 2:
                raise ValueError("image_dimensions must be (height, width) or a {width, height} mapping")
            height, width = value
        if (
            isinstance(height, bool)
            or isinstance(width, bool)
            or not isinstance(height, Integral)
            or not isinstance(width, Integral)
            or height <= 0
            or width <= 0
        ):
            raise ValueError("image dimensions must be positive integers in (height, width) order")
        return int(height), int(width)

    @staticmethod
    def _frame_id(value: Any) -> str:
        if isinstance(value, bool):
            raise ValueError("frame_id must be a non-empty string or non-negative integer")
        if isinstance(value, Integral) and value >= 0:
            return str(int(value))
        if isinstance(value, str) and value.strip():
            return value.strip()
        raise ValueError("frame_id must be a non-empty string or non-negative integer")

    @staticmethod
    def _frame_number(value: Any) -> int:
        if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
            raise ValueError("frame_number must be a positive integer (1-based)")
        return int(value)

    @staticmethod
    def _detections(value: Any) -> list[Mapping[str, Any]]:
        if not isinstance(value, (list, tuple)):
            raise ValueError("detections must be a list or tuple; use [] for a valid empty frame")
        if any(not isinstance(item, Mapping) for item in value):
            raise ValueError("detections must contain only object mappings")
        return list(value)

    @staticmethod
    def _graph_data(graph: nx.MultiDiGraph) -> dict[str, Any]:
        return {
            "frame_id": graph.graph.get("frame_id"),
            "nodes": get_objects(graph),
            "edges": [
                {
                    "source": source,
                    "target": target,
                    "key": key,
                    **dict(attributes),
                }
                for source, target, key, attributes in graph.edges(keys=True, data=True)
            ],
            "warnings": list(graph.graph.get("warnings", [])),
        }

    @staticmethod
    def _deduplicate_warnings(*groups: Any) -> list[str]:
        warnings: list[str] = []
        for group in groups:
            if isinstance(group, (list, tuple)):
                for warning in group:
                    message = str(warning)
                    if message not in warnings:
                        warnings.append(message)
        return warnings

    def _invalid_frame_result(
        self,
        frame_id: Any,
        frame_number: Any,
        image_dimensions: Any,
        detections: Any,
        error: Exception,
    ) -> dict[str, Any]:
        return {
            "frame": {
                "frame_id": frame_id,
                "frame_number": frame_number,
                "image_shape": image_dimensions,
            },
            "detections": deepcopy(detections) if isinstance(detections, (list, tuple)) else [],
            "spatial": None,
            "relationships": [],
            "memory": None,
            "graph": None,
            "scene_description": None,
            "warnings": [f"invalid frame input: {error}"],
            "status": "invalid_input",
        }

    def process_frame(
        self,
        *,
        frame_id: str | int,
        frame_number: int,
        image_dimensions: Sequence[int] | Mapping[str, Any],
        detections: Sequence[Mapping[str, Any]],
        depth_map: Any = None,
        timestamp: float | str | None = None,
        profiler: FrameProfiler | None = None,
    ) -> dict[str, Any]:
        """Process one caller-supplied synthetic frame through existing modules."""
        started_at = time.perf_counter() if profiler is not None else None
        try:
            normalized_frame_id = self._frame_id(frame_id)
            normalized_frame_number = self._frame_number(frame_number)
            image_shape = self._dimensions(image_dimensions)
            input_detections = self._detections(detections)
            if normalized_frame_id in self._seen_frame_ids:
                raise ValueError("frame_id must be unique for each processed frame")
            if self._latest_frame_number is not None and normalized_frame_number <= self._latest_frame_number:
                raise ValueError("frame_number must increase for each processed frame")
        except (TypeError, ValueError) as error:
            return self._invalid_frame_result(frame_id, frame_number, image_dimensions, detections, error)

        try:
            with profile_stage(profiler, "spatial_relationships"):
                spatial = calculate_spatial_relationships(
                    input_detections,
                    image_shape,
                    depth_map,
                    frame_id=normalized_frame_id,
                )
        except Exception as error:
            result = self._invalid_frame_result(
                normalized_frame_id,
                normalized_frame_number,
                image_shape,
                input_detections,
                error,
            )
            result["status"] = "error"
            result["warnings"] = [f"spatial processing failed: {type(error).__name__}: {error}"]
            return result

        try:
            staged_memory = deepcopy(self.scene_memory)
            with profile_stage(profiler, "scene_memory"):
                memory_snapshot = staged_memory.update(
                    spatial,
                    frame_number=normalized_frame_number,
                    timestamp=timestamp,
                )
            with profile_stage(profiler, "scene_graph"):
                graph = build_scene_graph(spatial)
                graph_data = self._graph_data(graph)
            with profile_stage(profiler, "scene_description"):
                scene_description = describe_scene(
                    spatial,
                    scene_graph=graph,
                    scene_memory=staged_memory,
                )
            with profile_stage(profiler, "scene_narration"):
                scene_summary_speech = prepare_spoken_scene_summary(scene_description)
            warnings = self._deduplicate_warnings(
                spatial.get("warnings", []),
                memory_snapshot.get("warnings", []),
                graph_data.get("warnings", []),
                scene_description.get("warnings", []),
            )
            status = "partial" if warnings else "success"
            frame_metadata = {
                "frame_id": normalized_frame_id,
                "frame_number": normalized_frame_number,
                "timestamp": timestamp,
                "image_shape": image_shape,
                "width": image_shape[1],
                "height": image_shape[0],
            }
            result = {
                "frame": frame_metadata,
                "detections": deepcopy(input_detections),
                "spatial": spatial,
                "objects": spatial["objects"],
                "relationships": spatial["relationships"],
                "memory": memory_snapshot,
                "graph": graph_data,
                "scene_description": scene_description,
                "scene_summary_speech": scene_summary_speech,
                "warnings": warnings,
                "status": status,
            }
        except Exception as error:
            return {
                "frame": {
                    "frame_id": normalized_frame_id,
                    "frame_number": normalized_frame_number,
                    "timestamp": timestamp,
                    "image_shape": image_shape,
                    "width": image_shape[1],
                    "height": image_shape[0],
                },
                "detections": deepcopy(input_detections),
                "spatial": spatial,
                "relationships": spatial["relationships"],
                "memory": None,
                "graph": None,
                "scene_description": None,
                "warnings": [f"downstream processing failed: {type(error).__name__}: {error}"],
                "status": "error",
            }

        self.scene_memory.__dict__ = staged_memory.__dict__
        self._latest_spatial = spatial
        self._latest_graph = graph
        self._latest_scene_description = scene_description
        self._latest_frame_id = normalized_frame_id
        self._latest_frame_number = normalized_frame_number
        self._latest_commit_wall_time = time.monotonic()
        self._seen_frame_ids.add(normalized_frame_id)
        if profiler is not None and started_at is not None:
            profiler.record("pipeline_orchestration_total", (time.perf_counter() - started_at) * 1000)
        return result

    def describe_current_scene(self) -> dict[str, Any]:
        """Return the latest committed scene description and its speech text."""
        if self._latest_scene_description is None:
            return {
                "scene_description": None,
                "speech_response": None,
                "status": "not_ready",
                "warnings": ["no valid frame has been processed"],
            }
        return {
            "frame_id": self._latest_frame_id,
            "frame_number": self._latest_frame_number,
            "scene_description": self._latest_scene_description,
            "speech_response": prepare_spoken_scene_summary(self._latest_scene_description),
            "status": "success",
            "warnings": list(self._latest_scene_description.get("warnings", [])),
        }

    def query(self, text: Any, *, scene_age_seconds: float | None = None) -> dict[str, Any]:
        """Answer a query against the latest spatial/graph snapshot and live memory."""
        if self._latest_spatial is None or self._latest_graph is None:
            speech_response = prepare_spoken_response({
                "status": "not_ready",
                "warnings": ["no valid frame has been processed"],
            })
            return {
                "query": text,
                "query_result": None,
                "speech_response": speech_response,
                "status": "not_ready",
                "warnings": ["no valid frame has been processed"],
            }

        query_result = answer_query(
            text,
            spatial_output=self._latest_spatial,
            scene_graph=self._latest_graph,
            scene_memory=self.scene_memory,
            scene_age_seconds=scene_age_seconds,
        )
        speech_response = prepare_spoken_response(query_result)
        warnings = self._deduplicate_warnings(query_result.get("warnings", []), speech_response.get("warnings", []))
        return {
            "frame_id": self._latest_frame_id,
            "frame_number": self._latest_frame_number,
            "query": text,
            "query_result": query_result,
            "speech_response": speech_response,
            "status": query_result["status"],
            "warnings": warnings,
        }