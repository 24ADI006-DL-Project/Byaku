from pathlib import Path
import math
import statistics
import time
from typing import Any

import cv2
import numpy as np

from .detection_tracking import DEFAULT_WEIGHTS, DetectionTrackingAdapter
from .depth_integration import MODEL_NAME, DepthAnythingV2Adapter
from .pipeline_orchestrator import PipelineOrchestrator


class OfflineVideoPipeline:
    """Process a local video sequentially through the existing BYAKU modules."""

    def __init__(
        self,
        *,
        weights_path: str | Path = DEFAULT_WEIGHTS,
        depth_model_name: str = MODEL_NAME,
        confidence_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        detection_adapter: Any = None,
        depth_adapter: Any = None,
        orchestrator: PipelineOrchestrator | None = None,
        capture_factory: Any = None,
    ) -> None:
        self.weights_path = Path(weights_path).expanduser().resolve()
        self.depth_model_name = depth_model_name
        self.confidence_threshold = float(confidence_threshold)
        self.iou_threshold = float(iou_threshold)
        self.orchestrator = orchestrator if orchestrator is not None else PipelineOrchestrator()
        self.capture_factory = capture_factory or cv2.VideoCapture
        self.initialization_warnings: list[str] = []

        if detection_adapter is None:
            try:
                detection_adapter = DetectionTrackingAdapter(
                    weights_path=self.weights_path,
                    confidence_threshold=self.confidence_threshold,
                    iou_threshold=self.iou_threshold,
                )
            except Exception as error:
                self.initialization_warnings.append(
                    f"YOLO/ByteTrack initialization failed: {type(error).__name__}: {error}"
                )
        self.detection_adapter = detection_adapter

        if depth_adapter is None:
            try:
                depth_adapter = DepthAnythingV2Adapter(model_name=depth_model_name)
            except Exception as error:
                self.initialization_warnings.append(
                    f"Depth Anything V2 initialization failed: {type(error).__name__}: {error}"
                )
        self.depth_adapter = depth_adapter

    @staticmethod
    def _timing_summary(samples: list[float]) -> dict[str, Any]:
        if not samples:
            return {"status": "not_measured", "total_ms": None, "average_ms": None, "samples": 0}
        return {
            "status": "measured",
            "total_ms": sum(samples),
            "average_ms": statistics.fmean(samples),
            "samples": len(samples),
        }

    @staticmethod
    def _not_isolated(stage: str) -> dict[str, Any]:
        return {
            "status": "not_isolated",
            "total_ms": None,
            "average_ms": None,
            "samples": 0,
            "reason": f"{stage} executes inside PipelineOrchestrator.process_frame and has no stage timer",
        }

    @staticmethod
    def _unique_warnings(warnings: list[str], new_warnings: Any) -> None:
        if not isinstance(new_warnings, (list, tuple)):
            return
        for warning in new_warnings:
            message = str(warning)
            if message not in warnings:
                warnings.append(message)

    @staticmethod
    def _time_seconds(capture: Any, property_id: int) -> float | None:
        try:
            value = float(capture.get(property_id))
        except Exception:
            return None
        return value / 1000.0 if math.isfinite(value) and value >= 0 else None

    @staticmethod
    def _capture_value(capture: Any, property_id: int) -> float | None:
        try:
            value = float(capture.get(property_id))
        except Exception:
            return None
        return value if math.isfinite(value) else None

    def _empty_result(
        self,
        video_path: Path,
        configuration: dict[str, Any],
        status: str,
        warnings: list[str],
        started_at: float,
        *,
        video_metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        elapsed_ms = (time.perf_counter() - started_at) * 1000
        return {
            "video": video_metadata or {"path": str(video_path), "readable": False},
            "configuration": configuration,
            "frames_read": 0,
            "frames_processed": 0,
            "frames_skipped": 0,
            "frames_failed": 0,
            "per_frame_results": [],
            "stage_timings_ms": {
                "detection_tracking": self._timing_summary([]),
                "depth_estimation": self._timing_summary([]),
                "orchestrator_combined": self._timing_summary([]),
                "spatial_processing": self._not_isolated("Spatial processing"),
                "graph_memory_updates": self._not_isolated("Graph/memory updates"),
                "scene_description": self._not_isolated("Scene description"),
            },
            "total_processing_time_ms": elapsed_ms,
            "average_processing_time_per_processed_frame_ms": None,
            "latest_scene_description": None,
            "query_result": None,
            "speech_response": None,
            "query_time_ms": None,
            "final_status": status,
            "warnings": warnings,
        }

    def process_video(
        self,
        video_path: str | Path,
        *,
        max_frames: int | None = None,
        frame_stride: int = 1,
        query: str | None = None,
    ) -> dict[str, Any]:
        """Process selected frames in source order without writing or displaying video."""
        started_at = time.perf_counter()
        path = Path(video_path).expanduser()
        configuration = {
            "max_frames": max_frames,
            "frame_stride": frame_stride,
            "confidence_threshold": self.confidence_threshold,
            "iou_threshold": self.iou_threshold,
            "depth_model": self.depth_model_name,
            "yolo_weights": str(self.weights_path),
        }

        if max_frames is not None and (
            isinstance(max_frames, bool) or not isinstance(max_frames, int) or max_frames < 1
        ):
            return self._empty_result(path, configuration, "invalid_input", ["max_frames must be a positive integer or None"], started_at)
        if isinstance(frame_stride, bool) or not isinstance(frame_stride, int) or frame_stride < 1:
            return self._empty_result(path, configuration, "invalid_input", ["frame_stride must be a positive integer"], started_at)
        if not path.is_file():
            return self._empty_result(path, configuration, "error", [f"video file does not exist: {path}"], started_at)
        if self.initialization_warnings or self.detection_adapter is None or self.depth_adapter is None:
            warnings = list(self.initialization_warnings)
            if self.detection_adapter is None and not warnings:
                warnings.append("YOLO/ByteTrack adapter is unavailable")
            if self.depth_adapter is None and not warnings:
                warnings.append("Depth Anything V2 adapter is unavailable")
            return self._empty_result(path, configuration, "error", warnings, started_at)

        warnings: list[str] = []
        frames_read = 0
        frames_processed = 0
        frames_skipped = 0
        frames_failed = 0
        selected_frames = 0
        per_frame_results: list[dict[str, Any]] = []
        detection_times: list[float] = []
        depth_times: list[float] = []
        orchestrator_times: list[float] = []
        capture = None
        video_metadata: dict[str, Any] = {"path": str(path), "readable": False}
        should_stop = False
        processing_completed_at: float | None = None

        try:
            try:
                capture = self.capture_factory(str(path))
            except Exception as error:
                warnings.append(f"could not initialize video reader: {type(error).__name__}: {error}")
                return self._empty_result(path, configuration, "error", warnings, started_at)

            try:
                opened = capture.isOpened()
            except Exception as error:
                warnings.append(f"could not inspect video reader: {type(error).__name__}: {error}")
                opened = False
            if not opened:
                warnings.append(f"could not open video: {path}")
                video_metadata["readable"] = False
                return self._empty_result(
                    path,
                    configuration,
                    "error",
                    warnings,
                    started_at,
                    video_metadata=video_metadata,
                )

            fps_value = self._capture_value(capture, cv2.CAP_PROP_FPS)
            count_value = self._capture_value(capture, cv2.CAP_PROP_FRAME_COUNT)
            width_value = self._capture_value(capture, cv2.CAP_PROP_FRAME_WIDTH)
            height_value = self._capture_value(capture, cv2.CAP_PROP_FRAME_HEIGHT)
            reported_frame_count = int(count_value) if count_value is not None and count_value > 0 else None
            width = int(width_value) if width_value is not None and width_value > 0 else None
            height = int(height_value) if height_value is not None and height_value > 0 else None
            video_metadata = {
                "path": str(path),
                "readable": True,
                "fps": fps_value if fps_value is not None and fps_value > 0 else None,
                "reported_frame_count": reported_frame_count,
                "width": width,
                "height": height,
            }
            if video_metadata["fps"] is None:
                warnings.append("video FPS metadata is unavailable")
            if reported_frame_count is None:
                warnings.append("reported video frame-count metadata is unavailable")

            while True:
                if max_frames is not None and selected_frames >= max_frames:
                    break
                try:
                    read_ok, frame = capture.read()
                except Exception as error:
                    frames_failed += 1
                    warnings.append(f"video read failed: {type(error).__name__}: {error}")
                    break
                if not read_ok or frame is None:
                    break

                frames_read += 1
                source_frame_number = frames_read
                if (source_frame_number - 1) % frame_stride != 0:
                    frames_skipped += 1
                    continue
                selected_frames += 1

                timestamp_seconds = self._time_seconds(capture, cv2.CAP_PROP_POS_MSEC)
                if timestamp_seconds is None:
                    warnings.append(f"timestamp unavailable for source frame {source_frame_number}")
                frame_id = f"{path.stem}:{source_frame_number}"
                frame_height, frame_width = frame.shape[:2]
                frame_started_at = time.perf_counter()

                detection_started_at = time.perf_counter()
                try:
                    detection_result = self.detection_adapter.process_frame(
                        frame,
                        frame_id=frame_id,
                        frame_number=source_frame_number,
                    )
                except Exception as error:
                    detection_result = {
                        "status": "error",
                        "warnings": [f"YOLO/ByteTrack failed: {type(error).__name__}: {error}"],
                    }
                detection_elapsed_ms = (time.perf_counter() - detection_started_at) * 1000
                detection_times.append(detection_elapsed_ms)
                self._unique_warnings(warnings, detection_result.get("warnings", []))
                if detection_result.get("status") not in {"success", "partial"}:
                    frames_failed += 1
                    per_frame_results.append(
                        {
                            "frame": {
                                "frame_id": frame_id,
                                "frame_number": source_frame_number,
                                "timestamp_seconds": timestamp_seconds,
                                "image_shape": (frame_height, frame_width),
                            },
                            "status": detection_result.get("status", "error"),
                            "detection_count": 0,
                            "track_count": 0,
                            "depth_status": "not_run",
                            "warnings": list(detection_result.get("warnings", [])),
                            "stage_timings_ms": {
                                "detection_tracking": detection_elapsed_ms,
                                "depth_estimation": None,
                                "orchestrator_combined": None,
                            },
                        }
                    )
                    should_stop = True
                    break

                depth_started_at = time.perf_counter()
                try:
                    depth_result = self.depth_adapter.estimate_depth(frame)
                except Exception as error:
                    depth_result = {
                        "status": "error",
                        "depth_available": False,
                        "depth_map": None,
                        "metadata": {},
                        "warnings": [f"Depth Anything V2 failed: {type(error).__name__}: {error}"],
                    }
                depth_elapsed_ms = (time.perf_counter() - depth_started_at) * 1000
                depth_times.append(depth_elapsed_ms)
                self._unique_warnings(warnings, depth_result.get("warnings", []))

                detection_frame = detection_result.get("frame", {})
                tracking_frame_shape = detection_frame.get("image_shape") if isinstance(detection_frame, dict) else None
                if (
                    detection_frame.get("frame_id") != frame_id
                    or detection_frame.get("frame_number") != source_frame_number
                    or tracking_frame_shape != (frame_height, frame_width)
                ):
                    frames_failed += 1
                    warnings.append(f"detection adapter returned mismatched metadata for source frame {source_frame_number}")
                    should_stop = True
                    break

                if depth_result.get("depth_available"):
                    depth_map = depth_result.get("depth_map")
                    if not isinstance(depth_map, np.ndarray) or depth_map.shape != (frame_height, frame_width) or not np.isfinite(depth_map).all():
                        warnings.append(f"depth adapter returned an invalid map for source frame {source_frame_number}")
                        depth_map = None
                        depth_status = "error"
                    else:
                        depth_status = depth_result.get("status", "success")
                else:
                    depth_map = None
                    depth_status = depth_result.get("status", "error")

                orchestrator_started_at = time.perf_counter()
                try:
                    pipeline_result = self.orchestrator.process_frame(
                        frame_id=frame_id,
                        frame_number=source_frame_number,
                        image_dimensions=(frame_height, frame_width),
                        detections=detection_result.get("orchestrator_detections", []),
                        depth_map=depth_map,
                        timestamp=timestamp_seconds,
                    )
                except Exception as error:
                    pipeline_result = {
                        "status": "error",
                        "warnings": [f"orchestrator failed: {type(error).__name__}: {error}"],
                    }
                orchestrator_elapsed_ms = (time.perf_counter() - orchestrator_started_at) * 1000
                orchestrator_times.append(orchestrator_elapsed_ms)
                self._unique_warnings(warnings, pipeline_result.get("warnings", []))

                if pipeline_result.get("status") not in {"success", "partial"}:
                    frames_failed += 1
                    should_stop = True
                else:
                    frames_processed += 1
                frame_elapsed_ms = (time.perf_counter() - frame_started_at) * 1000
                graph = pipeline_result.get("graph", {})
                memory = pipeline_result.get("memory", {})
                scene = pipeline_result.get("scene_description", {})
                per_frame_results.append(
                    {
                        "frame": {
                            "frame_id": frame_id,
                            "frame_number": source_frame_number,
                            "timestamp_seconds": timestamp_seconds,
                            "image_shape": (frame_height, frame_width),
                        },
                        "status": pipeline_result.get("status", "error"),
                        "detection_count": len(detection_result.get("detections", [])),
                        "track_count": len(detection_result.get("tracks", [])),
                        "untracked_detection_count": detection_result.get("untracked_detection_count", 0),
                        "detections": detection_result.get("detections", []),
                        "tracks": detection_result.get("tracks", []),
                        "depth": {
                            "status": depth_status,
                            "depth_available": depth_map is not None,
                            "metadata": depth_result.get("metadata", {}),
                        },
                        "objects": pipeline_result.get("objects", []),
                        "relationships": pipeline_result.get("relationships", []),
                        "graph": graph,
                        "memory": {
                            "visible_ids": [item.get("object_id") for item in memory.get("visible_objects", [])],
                            "missing_ids": [item.get("object_id") for item in memory.get("temporarily_missing_objects", [])],
                            "stale_ids": [item.get("object_id") for item in memory.get("stale_objects", [])],
                        },
                        "scene_description": scene,
                        "stage_timings_ms": {
                            "detection_tracking": detection_elapsed_ms,
                            "depth_estimation": depth_elapsed_ms,
                            "orchestrator_combined": orchestrator_elapsed_ms,
                            "frame_total": frame_elapsed_ms,
                        },
                    }
                )
                if should_stop:
                    break
        finally:
            if capture is not None:
                try:
                    capture.release()
                except Exception as error:
                    warnings.append(f"video release failed: {type(error).__name__}: {error}")
            processing_completed_at = time.perf_counter()

        total_processing_time_ms = (processing_completed_at - started_at) * 1000
        if frames_read == 0:
            self._unique_warnings(warnings, ["video opened but yielded no frames"])
        if selected_frames == 0 and frames_read > 0:
            warnings.append("frame sampling selected no frames")

        query_result = None
        speech_response = None
        query_time_ms = None
        latest_scene_description = None
        if frames_processed:
            latest_scene_description = self.orchestrator.describe_current_scene()
            if query is not None:
                query_started_at = time.perf_counter()
                query_result = self.orchestrator.query(query)
                query_time_ms = (time.perf_counter() - query_started_at) * 1000
                speech_response = query_result.get("speech_response")
                self._unique_warnings(warnings, query_result.get("warnings", []))

        stage_timings = {
            "detection_tracking": self._timing_summary(detection_times),
            "depth_estimation": self._timing_summary(depth_times),
            "orchestrator_combined": self._timing_summary(orchestrator_times),
            "spatial_processing": self._not_isolated("Spatial processing"),
            "graph_memory_updates": self._not_isolated("Graph/memory updates"),
            "scene_description": self._not_isolated("Scene description"),
        }
        if frames_processed == 0:
            final_status = "error"
        elif frames_failed or warnings or any(
            item.get("depth", {}).get("status") != "success" for item in per_frame_results
        ):
            final_status = "partial"
        else:
            final_status = "success"

        result = {
            "video": video_metadata,
            "configuration": configuration,
            "frames_read": frames_read,
            "frames_processed": frames_processed,
            "frames_skipped": frames_skipped,
            "frames_failed": frames_failed,
            "frame_limit_reached": max_frames is not None and selected_frames >= max_frames,
            "per_frame_results": per_frame_results,
            "stage_timings_ms": stage_timings,
            "total_processing_time_ms": total_processing_time_ms,
            "average_processing_time_per_processed_frame_ms": (
                total_processing_time_ms / frames_processed if frames_processed else None
            ),
            "latest_scene_description": latest_scene_description,
            "query_result": query_result,
            "speech_response": speech_response,
            "query_time_ms": query_time_ms,
            "final_status": final_status,
            "warnings": warnings,
        }
        return result