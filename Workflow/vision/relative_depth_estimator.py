from collections.abc import Mapping
import math
from pathlib import Path
import time
from typing import Any, Callable

import cv2
import numpy as np
import torch
import torch.nn.functional as functional
from PIL import Image

from Backend.frame_timing_profiler import FrameProfiler, profile_stage


MODEL_NAME = "depth-anything/Depth-Anything-V2-Small-hf"
DEPTH_INPUT_SIZE = 364


def normalize_relative_depth_map(depth_values: Any, frame_shape: tuple[int, int]) -> dict[str, Any]:
    """Validate, resize, and min-max normalize one raw relative-depth prediction.

    The local Depth Anything V2 Small checkpoint produced larger values for
    nearer foreground regions in probes of both project bedroom images. The
    returned normalized map preserves that ordering: higher values mean nearer.
    A constant map is treated as unavailable because it carries no relative
    separation evidence.
    """
    try:
        height, width = frame_shape
    except (TypeError, ValueError):
        return {
            "status": "invalid_input",
            "depth_map": None,
            "depth_available": False,
            "metadata": {},
            "warnings": ["frame_shape must be a (height, width) pair"],
        }
    if (
        isinstance(height, bool)
        or isinstance(width, bool)
        or not isinstance(height, (int, np.integer))
        or not isinstance(width, (int, np.integer))
        or height <= 0
        or width <= 0
    ):
        return {
            "status": "invalid_input",
            "depth_map": None,
            "depth_available": False,
            "metadata": {},
            "warnings": ["frame_shape dimensions must be positive integers"],
        }

    try:
        raw = np.asarray(depth_values, dtype=np.float32)
    except (TypeError, ValueError, OverflowError) as error:
        return {
            "status": "error",
            "depth_map": None,
            "depth_available": False,
            "metadata": {},
            "warnings": [f"depth output is not a numeric array: {error}"],
        }

    if raw.ndim == 3 and raw.shape[0] == 1:
        raw = raw[0]
    if raw.ndim != 2 or raw.size == 0:
        return {
            "status": "error",
            "depth_map": None,
            "depth_available": False,
            "metadata": {"raw_shape": tuple(raw.shape)},
            "warnings": ["depth output must be a non-empty 2D map (or a single-item batch)"],
        }
    if not np.isfinite(raw).all():
        return {
            "status": "error",
            "depth_map": None,
            "depth_available": False,
            "metadata": {"raw_shape": tuple(raw.shape)},
            "warnings": ["depth output contains NaN or infinite values"],
        }

    raw_shape = tuple(int(value) for value in raw.shape)
    if raw_shape != (height, width):
        raw = cv2.resize(raw, (int(width), int(height)), interpolation=cv2.INTER_CUBIC)
        raw = np.asarray(raw, dtype=np.float32)
    if raw.shape != (height, width) or not np.isfinite(raw).all():
        return {
            "status": "error",
            "depth_map": None,
            "depth_available": False,
            "metadata": {"raw_shape": raw_shape, "output_shape": tuple(raw.shape)},
            "warnings": ["resized depth map has an invalid shape or non-finite values"],
        }

    raw_min = float(raw.min())
    raw_max = float(raw.max())
    metadata = {
        "raw_shape": raw_shape,
        "output_shape": (int(height), int(width)),
        "raw_min": raw_min,
        "raw_max": raw_max,
        "normalization": "per-frame min-max to [0, 1]",
        "depth_ordering": "higher_is_nearer",
        "metric_distance_available": False,
    }
    if math.isclose(raw_min, raw_max, rel_tol=1e-6, abs_tol=1e-8):
        return {
            "status": "partial",
            "depth_map": None,
            "depth_available": False,
            "metadata": metadata,
            "warnings": ["depth prediction is constant; relative depth is unavailable for this frame"],
        }

    normalized = ((raw - raw_min) / (raw_max - raw_min)).astype(np.float32)
    if not np.isfinite(normalized).all():
        return {
            "status": "error",
            "depth_map": None,
            "depth_available": False,
            "metadata": metadata,
            "warnings": ["normalized depth map contains non-finite values"],
        }
    metadata["normalized_min"] = float(normalized.min())
    metadata["normalized_max"] = float(normalized.max())
    return {
        "status": "success",
        "depth_map": normalized,
        "depth_available": True,
        "metadata": metadata,
        "warnings": [],
    }


class DepthAnythingV2Adapter:
    """Load the cached Depth Anything V2 Small model once and adapt its output."""

    def __init__(
        self,
        model_name: str = MODEL_NAME,
        *,
        device: str | None = None,
        processor: Any = None,
        model: Any = None,
        processor_factory: Any = None,
        model_factory: Any = None,
    ) -> None:
        if not isinstance(model_name, str) or not model_name.strip():
            raise ValueError("model_name must be a non-empty string")
        if (processor is None) != (model is None):
            raise ValueError("processor and model must either both be supplied or both be loaded")

        self.model_name = model_name
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        if processor is not None and model is not None:
            self.processor = processor
            self.model = model
        else:
            try:
                if processor_factory is None or model_factory is None:
                    from transformers import AutoImageProcessor, AutoModelForDepthEstimation
                    processor_factory = processor_factory or AutoImageProcessor
                    model_factory = model_factory or AutoModelForDepthEstimation
                self.processor = processor_factory.from_pretrained(
                    model_name,
                    local_files_only=True,
                )
                loaded_model = model_factory.from_pretrained(
                    model_name,
                    local_files_only=True,
                )
                moved_model = loaded_model.to(self.device)
                self.model = moved_model if moved_model is not None else loaded_model
                self.model.eval()
                if self.device.type == "cpu":
                    current_threads = torch.get_num_threads()
                    if current_threads > 8:
                        torch.set_num_threads(8)
            except Exception as error:
                raise RuntimeError(
                    f"Could not load cached Depth Anything model {model_name!r} "
                    f"with downloads disabled: {type(error).__name__}: {error}"
                ) from error

    @staticmethod
    def _failure(status: str, message: str, frame_shape: tuple[int, int] | None = None) -> dict[str, Any]:
        return {
            "status": status,
            "depth_map": None,
            "depth_available": False,
            "metadata": {
                "input_shape": frame_shape,
                "depth_ordering": "higher_is_nearer",
                "metric_distance_available": False,
            },
            "inference_time_ms": None,
            "warnings": [message],
        }

    def estimate_depth(
        self,
        frame: Any,
        *,
        profiler: FrameProfiler | None = None,
    ) -> dict[str, Any]:
        """Return a finite frame-sized normalized relative-depth map or an explicit error."""
        if (
            not isinstance(frame, np.ndarray)
            or frame.ndim != 3
            or frame.shape[2] != 3
            or frame.size == 0
            or frame.dtype != np.uint8
        ):
            return self._failure("invalid_input", "frame must be a non-empty uint8 HxWx3 BGR image")
        height, width = frame.shape[:2]
        frame_shape = (height, width)
        try:
            with profile_stage(profiler, "depth_preprocessing"):
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                processor_size = {"height": DEPTH_INPUT_SIZE, "width": DEPTH_INPUT_SIZE}
                if profiler is not None:
                    profiler.add_metadata("depth_processor_source_image_shape", list(rgb_frame.shape))
                    profiler.add_metadata("depth_processor_size_argument", processor_size.copy())
                    profiler.add_metadata(
                        "depth_processor_keep_aspect_ratio",
                        getattr(self.processor, "keep_aspect_ratio", None),
                    )
                    profiler.add_metadata(
                        "depth_processor_ensure_multiple_of",
                        getattr(self.processor, "ensure_multiple_of", None),
                    )
                    profiler.add_metadata(
                        "depth_processor_do_resize",
                        getattr(self.processor, "do_resize", None),
                    )
                    profiler.add_metadata(
                        "depth_processor_do_pad",
                        getattr(self.processor, "do_pad", None),
                    )
                    profiler.add_metadata(
                        "depth_processor_crop_size",
                        getattr(self.processor, "crop_size", None),
                    )
                    profiler.add_metadata(
                        "depth_processor_class",
                        type(self.processor).__name__,
                    )
                inputs = self.processor(
                    images=rgb_frame,
                    return_tensors="pt",
                    size=processor_size,
                )
                if not isinstance(inputs, Mapping):
                    raise TypeError("image processor did not return a tensor mapping")
                inputs = {name: tensor.to(self.device) for name, tensor in inputs.items()}
                pixel_values = inputs.get("pixel_values")
                if profiler is not None and pixel_values is not None:
                    profiler.add_metadata("depth_processor_tensor_shape", list(pixel_values.shape))
                    profiler.add_metadata("depth_processor_input_size", DEPTH_INPUT_SIZE)
                    profiler.add_metadata("depth_model_input_tensor_shape", list(pixel_values.shape))
            started_at = time.perf_counter()
            with profile_stage(profiler, "depth_model_inference"):
                with torch.inference_mode():
                    output = self.model(**inputs)
                if self.device.type == "cuda":
                    torch.cuda.synchronize(self.device)
            inference_time_ms = (time.perf_counter() - started_at) * 1000
            raw_prediction = getattr(output, "predicted_depth", None)
            if raw_prediction is None:
                return self._failure("error", "depth model output has no predicted_depth field", frame_shape)
            if profiler is not None and hasattr(raw_prediction, "shape"):
                profiler.add_metadata("depth_model_output_tensor_shape", list(raw_prediction.shape))
            with profile_stage(profiler, "depth_postprocessing"):
                raw_array = _to_numpy(raw_prediction)
                normalized_result = normalize_relative_depth_map(raw_array, frame_shape)
            normalized_result["metadata"].update(
                {
                    "model": self.model_name,
                    "device": str(self.device),
                    "input_shape": frame_shape,
                }
            )
            normalized_result["inference_time_ms"] = inference_time_ms
            return normalized_result
        except Exception as error:
            return self._failure(
                "error",
                f"Depth inference failed: {type(error).__name__}: {error}",
                frame_shape,
            )

    def process_frame(
        self,
        frame: Any,
        *,
        detection_result: Any,
        orchestrator: Any,
    ) -> dict[str, Any]:
        """Combine one depth estimate and detection/tracking result in the orchestrator."""
        if not isinstance(detection_result, Mapping):
            return {
                "status": "invalid_input",
                "frame": None,
                "depth": None,
                "pipeline_result": None,
                "warnings": ["detection_result must be a mapping"],
            }
        if detection_result.get("status") not in {"success", "partial"}:
            return {
                "status": "error",
                "frame": detection_result.get("frame"),
                "depth": None,
                "pipeline_result": None,
                "warnings": ["detection/tracking result is not successful; orchestrator was not called"],
            }
        frame_metadata = detection_result.get("frame")
        detections = detection_result.get("orchestrator_detections")
        if not isinstance(frame_metadata, Mapping) or not isinstance(detections, (list, tuple)):
            return {
                "status": "invalid_input",
                "frame": frame_metadata,
                "depth": None,
                "pipeline_result": None,
                "warnings": ["detection result is missing frame metadata or orchestrator_detections"],
            }
        if not callable(getattr(orchestrator, "process_frame", None)):
            return {
                "status": "invalid_input",
                "frame": frame_metadata,
                "depth": None,
                "pipeline_result": None,
                "warnings": ["orchestrator must provide process_frame()"],
            }
        if not isinstance(frame, np.ndarray) or frame.ndim != 3 or frame.shape[2] != 3 or frame.size == 0:
            return {
                "status": "invalid_input",
                "frame": frame_metadata,
                "depth": None,
                "pipeline_result": None,
                "warnings": ["frame must be a non-empty HxWx3 image"],
            }

        frame_shape = tuple(frame.shape[:2])
        metadata_shape = frame_metadata.get("image_shape")
        if metadata_shape is not None and tuple(metadata_shape) != frame_shape:
            return {
                "status": "invalid_input",
                "frame": frame_metadata,
                "depth": None,
                "pipeline_result": None,
                "warnings": ["detection frame dimensions do not match the supplied image"],
            }

        depth_result = self.estimate_depth(frame)
        if depth_result["status"] == "invalid_input":
            return {
                "status": "invalid_input",
                "frame": frame_metadata,
                "depth": depth_result,
                "pipeline_result": None,
                "warnings": list(depth_result["warnings"]),
            }
        try:
            pipeline_result = orchestrator.process_frame(
                frame_id=frame_metadata.get("frame_id"),
                frame_number=frame_metadata.get("frame_number"),
                image_dimensions=frame_shape,
                detections=list(detections),
                depth_map=depth_result["depth_map"] if depth_result["depth_available"] else None,
            )
        except Exception as error:
            return {
                "status": "error",
                "frame": frame_metadata,
                "depth": depth_result,
                "pipeline_result": None,
                "warnings": [f"orchestrator processing failed: {type(error).__name__}: {error}"],
            }

        warnings = list(depth_result["warnings"])
        warnings.extend(message for message in pipeline_result.get("warnings", []) if message not in warnings)
        if depth_result["status"] != "success" or pipeline_result.get("status") != "success":
            status = "partial" if pipeline_result.get("status") not in {"error", "invalid_input"} else "error"
        else:
            status = "success"
        return {
            "frame": frame_metadata,
            "model": self.model_name,
            "depth": {
                "status": depth_result["status"],
                "depth_available": depth_result["depth_available"],
                "metadata": depth_result["metadata"],
                "inference_time_ms": depth_result["inference_time_ms"],
                "depth_map": depth_result["depth_map"],
            },
            "pipeline_result": pipeline_result,
            "warnings": warnings,
            "status": status,
        }


def _to_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    return np.asarray(value)