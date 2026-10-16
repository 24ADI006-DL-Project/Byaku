import base64
from contextlib import asynccontextmanager
from collections.abc import Mapping
from datetime import UTC, datetime
import hashlib
import json
import logging
from pathlib import Path
import tempfile
from threading import Lock
import time
from typing import Any
from uuid import uuid4

import cv2
import numpy as np
from fastapi import FastAPI, HTTPException, Request, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator
from starlette.concurrency import run_in_threadpool
from starlette.exceptions import HTTPException as StarletteHTTPException

from Workflow.reasoning.spatial_relation_engine import (
    DEPTH_CATEGORY_MID_THRESHOLD,
    DEPTH_CATEGORY_NEAR_THRESHOLD,
)
from Workflow.vision.object_detector_tracker import DEFAULT_WEIGHTS, DetectionTrackingAdapter
from Workflow.vision.relative_depth_estimator import DepthAnythingV2Adapter, MODEL_NAME
from Workflow.vision.annotation_renderer import encode_annotated_frame
from Workflow.scene_pipeline import PipelineOrchestrator

from .browser_message_adapter import (
    adapt_browser_frame_result,
    adapt_browser_query_result,
    dashboard_reasoning,
)
from .frame_timing_profiler import FrameProfiler, profile_stage
from .feedback_store import FeedbackStore

_camera_operation_lock = Lock()
_frame_processing_lock = Lock()
_websocket_sessions_lock = Lock()
_review_frame_cache_lock = Lock()
_review_frame_cache: dict[str, tuple[str, float]] = {}
_UPLOAD_IMAGE_MAX_BYTES = 12 * 1024 * 1024
_UPLOAD_VIDEO_MAX_BYTES = 100 * 1024 * 1024
_UPLOAD_IMAGE_MAX_PIXELS = 40_000_000
_UPLOAD_VIDEO_MAX_FRAMES = 20
_UPLOAD_VIDEO_FRAME_STRIDE = 30
_UPLOAD_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}
_UPLOAD_VIDEO_SUFFIXES = {".mp4", ".mov", ".avi", ".mkv", ".webm"}
_logger = logging.getLogger(__name__)
_feedback_store = FeedbackStore(Path(__file__).resolve().parents[1] / "Data" / "feedback" / "review_history.json")
_FEEDBACK_CORRECTION_TYPES = {
    "confirm",
    "accept",
    "skip",
    "class_correction",
    "false_positive",
    "position_flag",
    "position_correction",
}


@asynccontextmanager
async def _api_lifespan(application: FastAPI):
    yield
    producer = getattr(application.state, "camera_producer", None)
    if producer is not None:
        producer.stop()
    application.state.camera_producer = None
    with _websocket_sessions_lock:
        _websocket_sessions().clear()


app = FastAPI(title="BYAKU Rebuild API", lifespan=_api_lifespan)


class FrameRequest(BaseModel):
    frame_id: str = Field(..., min_length=1)
    frame_number: int = Field(..., gt=0)
    width: int = Field(..., gt=0)
    height: int = Field(..., gt=0)
    data: str = Field(..., min_length=1)
    timestamp: float | None = None

    @field_validator("frame_id")
    @classmethod
    def _frame_id_must_be_non_empty(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("frame_id must be a non-empty string")
        return cleaned


class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1)

    @field_validator("query")
    @classmethod
    def _query_must_be_non_empty(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("query must be a non-empty string")
        return cleaned


class CameraStartRequest(BaseModel):
    source: int | str = 0


class FeedbackRequest(BaseModel):
    frame_id: str | None = None
    frame_number: int | None = Field(default=None, ge=0)
    review_frame_token: str | None = None
    session_id: str | None = None
    source: str | None = Field(default="live", max_length=50)
    detection_id: str | None = Field(default=None, max_length=120)
    submission_id: str = Field(default_factory=lambda: uuid4().hex, min_length=1, max_length=100)
    image_dimensions: dict[str, int] | None = None
    track_id: int | None = Field(default=None, ge=0)
    original_prediction: dict[str, Any]
    corrected_value: str | dict[str, Any] | None = None
    human_corrected_name: str | None = Field(default=None, min_length=1, max_length=80)
    correction_type: str
    review_status: str

    @field_validator("frame_id")
    @classmethod
    def _frame_id_must_be_non_empty(cls, value: str | None) -> str | None:
        if value is None:
            return None
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("frame_id must be non-empty when provided")
        return cleaned

    @field_validator("submission_id")
    @classmethod
    def _submission_id_must_be_non_empty(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("submission_id must be a non-empty string")
        return cleaned

    @field_validator("image_dimensions")
    @classmethod
    def _image_dimensions_must_be_positive(cls, value: dict[str, int] | None) -> dict[str, int] | None:
        if value is None:
            return None
        if set(value) != {"width", "height"} or any(
            isinstance(dimension, bool) or dimension <= 0 for dimension in value.values()
        ):
            raise ValueError("image_dimensions must contain positive width and height")
        return value

    @field_validator("correction_type")
    @classmethod
    def _correction_type_is_supported(cls, value: str) -> str:
        if value not in _FEEDBACK_CORRECTION_TYPES:
            raise ValueError("unsupported correction_type")
        return value

    @field_validator("review_status")
    @classmethod
    def _review_status_is_supported(cls, value: str) -> str:
        if value not in {"accepted", "confirmed", "skipped", "corrected"}:
            raise ValueError("review_status must be accepted, confirmed, skipped, or corrected")
        return value

    @field_validator("original_prediction")
    @classmethod
    def _original_prediction_has_class(cls, value: dict[str, Any]) -> dict[str, Any]:
        if not isinstance(value.get("class_name"), str) or not value["class_name"].strip():
            raise ValueError("original_prediction must include a non-empty class_name")
        return value

    @field_validator("human_corrected_name")
    @classmethod
    def _human_corrected_name_is_clean(cls, value: str | None) -> str | None:
        if value is None:
            return None
        cleaned = " ".join(value.split())
        if not cleaned or len(cleaned) > 80:
            raise ValueError("human_corrected_name must contain 1 to 80 non-empty characters")
        return cleaned


def _structured_error(
    *,
    status_code: int,
    code: str,
    message: str,
    recoverable: bool = True,
    stage: str = "api",
    details: Any | None = None,
) -> JSONResponse:
    content: dict[str, Any] = {
        "type": "error",
        "stage": stage,
        "code": code,
        "message": message,
        "recoverable": recoverable,
    }
    if details is not None:
        content["details"] = details
    return JSONResponse(status_code=status_code, content=content)


def _camera_error(
    *,
    status_code: int,
    code: str,
    message: str,
    recoverable: bool,
) -> JSONResponse:
    return _structured_error(
        status_code=status_code,
        code=code,
        message=message,
        recoverable=recoverable,
        stage="camera",
    )


def _new_camera_producer(source: int | str) -> Any:
    from .live_camera_producer import LiveCameraProducer

    capture_factory = getattr(app.state, "camera_capture_factory", None)
    return LiveCameraProducer(source, capture_factory=capture_factory)


def _websocket_sessions() -> dict[str, dict[str, Any]]:
    sessions = getattr(app.state, "websocket_sessions", None)
    if not isinstance(sessions, dict):
        sessions = {}
        app.state.websocket_sessions = sessions
    return sessions


def _browser_camera_connected() -> bool:
    with _websocket_sessions_lock:
        return any(bool(session.get("camera_active")) for session in _websocket_sessions().values())


def _python_camera_running() -> bool:
    producer = getattr(app.state, "camera_producer", None)
    return bool(producer is not None and getattr(producer, "is_running", False))


def _component_status(value: Any) -> str:
    if value in {"success", "ready"}:
        return "ready"
    if value in {"partial", "warning"}:
        return "partial"
    if value in {"error", "invalid_input"}:
        return "error"
    return "unavailable"


def _latest_stage_status() -> dict[str, str]:
    current = getattr(app.state, "latest_stage_status", None)
    if isinstance(current, Mapping):
        return dict(current)
    return {
        "detection": "unavailable",
        "tracking": "unavailable",
        "depth": "unavailable",
        "spatial": "unavailable",
        "scene_graph": "unavailable",
        "reasoning": "unavailable",
        "narration": "unavailable",
        "tts": "unavailable",
    }


def _stage_status(
    *,
    detection_result: Mapping[str, Any] | None = None,
    depth_result: Mapping[str, Any] | None = None,
    pipeline_result: Mapping[str, Any] | None = None,
) -> dict[str, str]:
    pipeline_status = pipeline_result.get("status") if pipeline_result else None
    graph_available = bool(pipeline_result and pipeline_result.get("graph") is not None)
    description_available = bool(pipeline_result and pipeline_result.get("scene_description") is not None)
    has_speech = bool(pipeline_result and pipeline_result.get("scene_summary_speech"))
    detection_status = detection_result.get("status") if detection_result else None
    depth_status = depth_result.get("status") if depth_result else None
    active_camera = _browser_camera_connected() or _python_camera_running()
    reasoning_status = _latest_stage_status().get("reasoning", "unavailable")
    return {
        "camera": "connected" if active_camera else "disconnected",
        "detection": _component_status(detection_status),
        "tracking": _component_status(detection_status),
        "depth": _component_status(depth_status),
        "spatial": _component_status(pipeline_status),
        "scene_graph": "ready" if graph_available else "unavailable",
        "reasoning": reasoning_status,
        "narration": "ready" if has_speech else "unavailable",
        "tts": "unavailable",
    }


def _next_websocket_frame_number() -> int:
    orchestrator = getattr(app.state, "orchestrator", None)
    latest_committed = getattr(orchestrator, "_latest_frame_number", None)
    last_assigned = getattr(app.state, "last_assigned_frame_number", 0)
    committed_number = latest_committed if isinstance(latest_committed, int) else 0
    assigned_number = last_assigned if isinstance(last_assigned, int) else 0
    next_number = max(committed_number, assigned_number) + 1
    app.state.last_assigned_frame_number = next_number
    return next_number


def _websocket_error(
    code: str,
    message: str,
    *,
    stage: str = "camera",
    recoverable: bool = True,
    frame_id: int | None = None,
) -> dict[str, Any]:
    error: dict[str, Any] = {
        "type": "error",
        "stage": stage,
        "code": code,
        "message": message,
        "recoverable": recoverable,
    }
    if frame_id is not None:
        error["frame_id"] = frame_id
    return error


async def _send_profiled_frame_result(
    websocket: WebSocket,
    payload: dict[str, Any],
    profiler: FrameProfiler | None,
) -> None:
    if profiler is None:
        await websocket.send_json(payload)
        return
    with profiler.measure("response_json_serialization"):
        serialized = json.dumps(payload, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
    with profiler.measure("websocket_send"):
        await websocket.send_text(serialized)


@app.exception_handler(StarletteHTTPException)
async def handle_http_exception(
    _request: Request,
    exception: StarletteHTTPException,
) -> JSONResponse:
    error_codes = {404: "not_found", 405: "method_not_allowed"}
    message = exception.detail if isinstance(exception.detail, str) else "Request failed"
    return _structured_error(
        status_code=exception.status_code,
        code=error_codes.get(exception.status_code, "http_error"),
        message=message,
        recoverable=400 <= exception.status_code < 500,
    )


@app.exception_handler(RequestValidationError)
async def handle_validation_exception(
    _request: Request,
    exception: RequestValidationError,
) -> JSONResponse:
    # exception.errors() may contain non-JSON-serializable values (e.g. the
    # original ValueError object) when a field_validator raises ValueError.
    # Sanitize each error record to ensure JSON serializability.
    try:
        raw_errors = getattr(exception, "errors")(include_url=False)
    except TypeError:
        raw_errors = exception.errors()
    safe_errors: list[dict] = []
    for err in raw_errors:
        safe_err = {}
        for k, v in err.items():
            if k == "ctx":
                # ctx often holds the raw exception; convert to str
                safe_ctx: dict = {}
                for ck, cv in (v.items() if isinstance(v, dict) else {}.items()):
                    safe_ctx[ck] = str(cv) if not isinstance(cv, (str, int, float, bool, type(None))) else cv
                safe_err["ctx"] = safe_ctx
            elif isinstance(v, (str, int, float, bool, list, dict, type(None))):
                safe_err[k] = v
            else:
                safe_err[k] = str(v)
        safe_errors.append(safe_err)
    first_error = safe_errors[0] if safe_errors else {}
    message = str(first_error.get("msg", "Request validation failed"))
    return _structured_error(
        status_code=422,
        code="invalid_request",
        message=message,
        recoverable=True,
        details=safe_errors,
    )


def _ensure_live_adapters() -> tuple[DetectionTrackingAdapter, DepthAnythingV2Adapter, PipelineOrchestrator]:
    if not hasattr(app.state, "orchestrator") or app.state.orchestrator is None:
        app.state.orchestrator = PipelineOrchestrator()
    if not hasattr(app.state, "detection_adapter") or app.state.detection_adapter is None:
        app.state.detection_adapter = DetectionTrackingAdapter(weights_path=DEFAULT_WEIGHTS)
    if not hasattr(app.state, "depth_adapter") or app.state.depth_adapter is None:
        app.state.depth_adapter = DepthAnythingV2Adapter(model_name=MODEL_NAME)
    return app.state.detection_adapter, app.state.depth_adapter, app.state.orchestrator


def _decode_base64_jpeg(frame_data: str) -> np.ndarray:
    encoded = frame_data.strip()
    if "data:" in encoded and "," in encoded:
        _, encoded = encoded.split(",", 1)
    try:
        raw = base64.b64decode(encoded, validate=True)
    except Exception as error:  # pragma: no cover - exercised through API tests
        raise HTTPException(status_code=400, detail=f"frame data is not valid base64 JPEG: {error}") from error
    buffer = np.frombuffer(raw, dtype=np.uint8)
    frame = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
    if frame is None or frame.size == 0 or frame.ndim != 3 or frame.shape[2] != 3:
        raise HTTPException(status_code=400, detail="frame data did not decode to a valid RGB/BGR image")
    return frame


def _encode_depth_preview(depth_map: np.ndarray | None) -> str | None:
    if depth_map is None:
        return None

    height, width = depth_map.shape
    scale = min(1.0, 640 / max(height, width))
    if scale < 1.0:
        preview_depth = cv2.resize(
            depth_map,
            (max(1, round(width * scale)), max(1, round(height * scale))),
            interpolation=cv2.INTER_AREA,
        )
    else:
        preview_depth = depth_map
    depth_values = np.clip(preview_depth, 0.0, 1.0)
    colorized_depth = np.empty((*depth_values.shape, 3), dtype=np.uint8)
    colorized_depth[depth_values < DEPTH_CATEGORY_MID_THRESHOLD] = (255, 0, 0)
    mid_range = (
        (depth_values >= DEPTH_CATEGORY_MID_THRESHOLD)
        & (depth_values < DEPTH_CATEGORY_NEAR_THRESHOLD)
    )
    colorized_depth[mid_range] = (0, 255, 255)
    colorized_depth[depth_values >= DEPTH_CATEGORY_NEAR_THRESHOLD] = (0, 0, 255)
    success, encoded = cv2.imencode(
        ".jpg",
        colorized_depth,
        [cv2.IMWRITE_JPEG_QUALITY, 80],
    )
    if not success:
        return None
    return base64.b64encode(encoded.tobytes()).decode("ascii")


def _serialize_frame_result(
    result: Mapping[str, Any],
    *,
    request: FrameRequest,
    depth_map: np.ndarray | None = None,
) -> dict[str, Any]:
    result_copy = dict(result)
    frame = result_copy.get("frame", {})
    frame_size = {"width": request.width, "height": request.height}
    response = {
        "type": "frame_result",
        "frame_id": frame.get("frame_id", request.frame_id),
        "frame_number": frame.get("frame_number", request.frame_number),
        "frame_size": frame_size,
        "image_shape": list(frame.get("image_shape", (request.height, request.width))),
        "status": result_copy.get("status", "error"),
        "warnings": list(result_copy.get("warnings", [])),
    }
    for key in ("detections", "objects", "relationships", "graph", "memory", "scene_description", "scene_summary_speech"):
        if key in result_copy:
            response[key] = result_copy[key]
    depth_preview = _encode_depth_preview(depth_map)
    if depth_preview is not None:
        response["depth_preview"] = depth_preview
    return response


def _validate_frame_identity(orchestrator: PipelineOrchestrator, *, frame_id: str, frame_number: int) -> None:
    seen_ids = getattr(orchestrator, "_seen_frame_ids", set())
    last_frame_number = getattr(orchestrator, "_latest_frame_number", None)
    if frame_id in seen_ids:
        raise HTTPException(status_code=409, detail=f"frame_id '{frame_id}' has already been processed")
    if last_frame_number is not None and frame_number <= last_frame_number:
        raise HTTPException(
            status_code=409,
            detail=(
                f"frame_number {frame_number} is not greater than the latest processed frame_number "
                f"{last_frame_number}"
            ),
        )


def _validated_depth_map(depth_result: Any, *, frame_shape: tuple[int, int]) -> tuple[np.ndarray | None, list[str]]:
    if not isinstance(depth_result, Mapping):
        raise HTTPException(status_code=400, detail="depth inference returned an invalid result")

    status = depth_result.get("status")
    warnings = [str(item) for item in depth_result.get("warnings", [])]
    depth_available = depth_result.get("depth_available")
    depth_map = depth_result.get("depth_map")
    if status == "partial" and depth_available is False and depth_map is None:
        metadata = depth_result.get("metadata")
        metadata = metadata if isinstance(metadata, Mapping) else {}
        raw_min = metadata.get("raw_min")
        raw_max = metadata.get("raw_max")
        constant_map = (
            isinstance(raw_min, (int, float))
            and isinstance(raw_max, (int, float))
            and np.isfinite([raw_min, raw_max]).all()
            and np.isclose(raw_min, raw_max, rtol=1e-6, atol=1e-8)
            and any("depth prediction is constant" in warning for warning in warnings)
        )
        if constant_map:
            return None, warnings

    if status != "success" or depth_available is not True:
        message = warnings[0] if warnings else "depth inference did not produce a usable depth map"
        raise HTTPException(status_code=400, detail=f"depth inference failed: {message}")

    try:
        validated_map = np.asarray(depth_map, dtype=np.float32)
    except (TypeError, ValueError, OverflowError) as error:
        raise HTTPException(status_code=400, detail=f"depth inference returned an invalid depth map: {error}") from error
    if validated_map.shape != frame_shape:
        raise HTTPException(
            status_code=400,
            detail=f"depth map dimensions {validated_map.shape} do not match frame dimensions {frame_shape}",
        )
    if not np.isfinite(validated_map).all():
        raise HTTPException(status_code=400, detail="depth map contains non-finite values")
    return validated_map, warnings


def _visual_fingerprint(frame: np.ndarray, bounding_box: Any) -> str | None:
    try:
        coordinates = np.asarray(bounding_box, dtype=np.float64)
    except (TypeError, ValueError, OverflowError):
        return None
    if coordinates.shape != (4,) or not np.isfinite(coordinates).all():
        return None
    x1, y1, x2, y2 = (int(round(value)) for value in coordinates)
    height, width = frame.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(width, x2), min(height, y2)
    if x2 <= x1 or y2 <= y1:
        return None
    normalized_crop = cv2.resize(frame[y1:y2, x1:x2], (32, 32), interpolation=cv2.INTER_AREA)
    return hashlib.sha256(normalized_crop.tobytes()).hexdigest()


def _correction_detection_identity(item: Mapping[str, Any]) -> tuple[str, Any] | None:
    track_id = item.get("track_id")
    if track_id is not None:
        return ("track", str(track_id))
    try:
        bounding_box = tuple(float(value) for value in item.get("bounding_box", ()))
    except (TypeError, ValueError, OverflowError):
        return None
    if len(bounding_box) != 4 or not np.isfinite(bounding_box).all():
        return None
    return ("box", bounding_box)


def _apply_remembered_corrections(
    frame: np.ndarray,
    detections: Any,
    *,
    warnings: list[str],
) -> tuple[dict[tuple[str, Any], dict[str, Any]], list[dict[str, Any]]]:
    metadata_by_identity: dict[tuple[str, Any], dict[str, Any]] = {}
    corrections_to_mark: list[dict[str, Any]] = []
    used_correction_ids: set[str] = set()
    if not isinstance(detections, list):
        return metadata_by_identity, corrections_to_mark
    for item in detections:
        if not isinstance(item, dict):
            continue
        predicted_class = item.get("class_name")
        if not isinstance(predicted_class, str):
            continue
        fingerprint = _visual_fingerprint(frame, item.get("bounding_box"))
        if fingerprint is None:
            continue
        try:
            correction = _feedback_store.find_exact_correction(
                predicted_class=predicted_class,
                visual_fingerprint=fingerprint,
            )
        except (OSError, ValueError, json.JSONDecodeError) as error:
            warnings.append(f"correction memory could not be checked: {error}")
            break
        if correction is not None:
            correction_id = correction.get("correction_id", correction.get("feedback_id"))
            if not isinstance(correction_id, str) or correction_id in used_correction_ids:
                continue
            identity = _correction_detection_identity(item)
            if identity is None:
                continue
            used_correction_ids.add(correction_id)
            metadata = {
                "correction_id": correction_id,
                "predicted_class": predicted_class,
                "corrected_name": correction["human_corrected_name"],
                "match_method": "exact_normalized_crop_sha256",
                "applied": False,
            }
            item["model_class_name"] = predicted_class
            item["class_name"] = correction["human_corrected_name"]
            item["remembered_correction"] = metadata
            metadata_by_identity[identity] = metadata
            corrections_to_mark.append(metadata)
    return metadata_by_identity, corrections_to_mark


def _attach_remembered_correction_metadata(
    objects: Any,
    metadata_by_identity: Mapping[tuple[str, Any], dict[str, Any]],
) -> None:
    if not isinstance(objects, list):
        return
    for item in objects:
        if not isinstance(item, dict):
            continue
        identity = _correction_detection_identity(item)
        metadata = metadata_by_identity.get(identity) if identity is not None else None
        if metadata is not None:
            item["model_class_name"] = metadata["predicted_class"]
            item["remembered_correction"] = dict(metadata)


def _snapshot_detection_identity(adapter: Any) -> tuple[set[str] | None, Any]:
    seen_ids = getattr(adapter, "_seen_frame_ids", None)
    last_frame_number = getattr(adapter, "_last_frame_number", None)
    return (set(seen_ids) if isinstance(seen_ids, set) else None, last_frame_number)


def _restore_detection_identity(adapter: Any, snapshot: tuple[set[str] | None, Any]) -> None:
    seen_ids, last_frame_number = snapshot
    if seen_ids is not None:
        adapter._seen_frame_ids = seen_ids
    if hasattr(adapter, "_last_frame_number"):
        adapter._last_frame_number = last_frame_number


@app.get("/health")
def health() -> dict[str, str]:
    return {"service": "byaku-rebuild-api", "status": "running"}


@app.get("/")
def root() -> dict[str, str]:
    return {
        "service": "BYAKU Rebuild API",
        "description": "Live frames, scene queries, image and video uploads, and camera capture API.",
    }


def _validate_upload_suffix(upload: UploadFile, allowed_suffixes: set[str], media_name: str) -> str:
    filename = upload.filename
    suffix = Path(filename).suffix.lower() if isinstance(filename, str) else ""
    if not suffix:
        raise HTTPException(status_code=400, detail=f"Choose a {media_name} file with a supported extension.")
    if suffix not in allowed_suffixes:
        supported = ", ".join(sorted(allowed_suffixes))
        raise HTTPException(status_code=415, detail=f"Unsupported {media_name} format. Supported extensions: {supported}.")
    return suffix


def _upload_size_label(maximum_bytes: int) -> str:
    mebibyte = 1024 * 1024
    if maximum_bytes >= mebibyte and maximum_bytes % mebibyte == 0:
        return f"{maximum_bytes // mebibyte} MB"
    return f"{maximum_bytes / mebibyte:g} MB" if maximum_bytes >= mebibyte else f"{maximum_bytes} bytes"


async def _read_bounded_upload(upload: UploadFile, maximum_bytes: int, media_name: str) -> bytes:
    chunks: list[bytes] = []
    total_bytes = 0
    while chunk := await upload.read(1024 * 1024):
        total_bytes += len(chunk)
        if total_bytes > maximum_bytes:
            raise HTTPException(
                status_code=413,
                detail=f"The {media_name} exceeds the {_upload_size_label(maximum_bytes)} upload limit.",
            )
        chunks.append(chunk)
    if total_bytes == 0:
        raise HTTPException(status_code=400, detail=f"The selected {media_name} file is empty.")
    return b"".join(chunks)


def _draw_upload_annotations(frame: np.ndarray, objects: list[dict[str, Any]]) -> str:
    try:
        encoded, _ = encode_annotated_frame(frame, objects)
    except (TypeError, ValueError, cv2.error) as error:
        raise HTTPException(status_code=500, detail="The annotated image could not be encoded.") from error
    return encoded


def _cache_review_frame(token: str, frame_jpeg_base64: str) -> None:
    now_monotonic = time.monotonic()
    with _review_frame_cache_lock:
        _review_frame_cache[token] = (frame_jpeg_base64, now_monotonic)
        expired = [
            cache_token for cache_token, (_, captured_at) in _review_frame_cache.items()
            if now_monotonic - captured_at > 180.0
        ]
        for cache_token in expired:
            _review_frame_cache.pop(cache_token, None)
        while len(_review_frame_cache) > 64:
            _review_frame_cache.pop(next(iter(_review_frame_cache)))


def _process_uploaded_image(frame: np.ndarray) -> dict[str, Any]:
    factory: Any = getattr(app.state, "upload_image_components_factory", None)
    if callable(factory):
        components: Any = factory()
        detection_adapter, depth_adapter, orchestrator = components
    else:
        detection_adapter = DetectionTrackingAdapter(weights_path=DEFAULT_WEIGHTS)
        depth_adapter = DepthAnythingV2Adapter(model_name=MODEL_NAME)
        orchestrator = PipelineOrchestrator()

    session_id = f"upload-image-{uuid4().hex[:12]}"
    frame_id = f"{session_id}:1"
    detection_result = detection_adapter.process_frame(frame, frame_id=frame_id, frame_number=1)
    if detection_result.get("status") not in {"success", "partial"}:
        raise HTTPException(status_code=422, detail="BYAKU could not detect objects in this image.")

    warnings: list[str] = []
    depth_map = None
    try:
        depth_result = depth_adapter.estimate_depth(frame)
        depth_map, warnings = _validated_depth_map(depth_result, frame_shape=frame.shape[:2])
    except Exception as error:
        warnings.append("Relative depth is unavailable; depth categories are shown as UNKNOWN.")
        _logger.warning("Depth inference failed for an uploaded image: %s", error)

    height, width = frame.shape[:2]
    pipeline_result = orchestrator.process_frame(
        frame_id=frame_id,
        frame_number=1,
        image_dimensions={"width": width, "height": height},
        detections=detection_result.get("orchestrator_detections", []),
        depth_map=depth_map,
    )
    if pipeline_result.get("status") not in {"success", "partial"}:
        raise HTTPException(status_code=422, detail="BYAKU could not complete scene analysis for this image.")
    raw_objects = pipeline_result.get("objects", [])
    objects: list[dict[str, Any]] = []
    for idx, item in enumerate(raw_objects):
        track_id = item.get("track_id")
        det_id = f"{session_id}:track:{track_id}" if track_id is not None else f"{session_id}:frame-1:det-{idx}"
        obj = dict(item)
        obj["detection_id"] = det_id
        obj["depth_category"] = obj.get("depth_category") or "UNKNOWN"
        obj["horizontal_position"] = obj.get("horizontal_position") or "UNKNOWN"
        objects.append(obj)

    scene = pipeline_result.get("scene_description")
    scene = scene if isinstance(scene, Mapping) else {}
    annotated_img = _draw_upload_annotations(frame, objects)

    review_frame_token = uuid4().hex
    raw_success, raw_encoded = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 88])
    if raw_success:
        _cache_review_frame(review_frame_token, base64.b64encode(raw_encoded.tobytes()).decode("ascii"))

    return {
        "type": "upload_result",
        "media_type": "image",
        "status": "partial" if warnings else pipeline_result.get("status", "success"),
        "session_id": session_id,
        "review_frame_token": review_frame_token,
        "annotated_image": annotated_img,
        "frame_size": {"width": width, "height": height},
        "objects": objects,
        "detection_count": len(objects),
        "scene_description": scene.get("description", ""),
        "warnings": warnings,
        "frames_processed": 1,
        "frames_failed": 0,
    }


async def _store_video_upload(upload: UploadFile, suffix: str, temporary_directory: Path) -> Path:
    upload_path = temporary_directory / f"video{suffix}"
    with upload_path.open("wb") as output:
        total_bytes = 0
        while chunk := await upload.read(1024 * 1024):
            total_bytes += len(chunk)
            if total_bytes > _UPLOAD_VIDEO_MAX_BYTES:
                raise HTTPException(
                    status_code=413,
                    detail=f"The video exceeds the {_upload_size_label(_UPLOAD_VIDEO_MAX_BYTES)} upload limit.",
                )
            output.write(chunk)
    if total_bytes == 0:
        raise HTTPException(status_code=400, detail="The selected video file is empty.")
    return upload_path


def _process_uploaded_video(video_path: Path) -> dict[str, Any]:
    factory: Any = getattr(app.state, "upload_video_pipeline_factory", None)
    if callable(factory):
        pipeline: Any = factory()
    else:
        from Workflow.tools.process_video_pipeline import OfflineVideoPipeline

        pipeline = OfflineVideoPipeline()
    result = pipeline.process_video(
        video_path,
        max_frames=_UPLOAD_VIDEO_MAX_FRAMES,
        frame_stride=_UPLOAD_VIDEO_FRAME_STRIDE,
    )
    if result.get("final_status") == "error" or not result.get("frames_processed"):
        processing_warnings = [str(warning).lower() for warning in result.get("warnings", [])]
        if any("initialization failed" in warning or "adapter is unavailable" in warning for warning in processing_warnings):
            raise HTTPException(
                status_code=503,
                detail="BYAKU's local video analysis models could not be initialized.",
            )
        if any("could not open video" in warning or "yielded no frames" in warning for warning in processing_warnings):
            raise HTTPException(
                status_code=422,
                detail="This video could not be decoded. Try a supported codec in MP4, MOV, AVI, MKV, or WebM.",
            )
        raise HTTPException(
            status_code=422,
            detail="BYAKU could not complete video analysis. Check the video file and try again.",
        )

    session_id = f"upload-video-{uuid4().hex[:12]}"
    samples: list[dict[str, Any]] = []
    unique_objects: dict[tuple[str, str], dict[str, Any]] = {}
    distinct_descriptions: list[str] = []
    for frame_result in result.get("per_frame_results", []):
        frame = frame_result.get("frame", {})
        frame_number = frame.get("frame_number")
        timestamp_seconds = frame.get("timestamp_seconds")
        frame_status = frame_result.get("status", "success")
        raw_objects = frame_result.get("objects", [])
        scene = frame_result.get("scene_description", {})
        description = scene.get("description") if isinstance(scene, Mapping) else ""
        if isinstance(description, str) and description.strip() and description not in distinct_descriptions:
            distinct_descriptions.append(description.strip())
        sample_objects: list[dict[str, Any]] = []
        for idx, item in enumerate(raw_objects):
            track_id = item.get("track_id")
            det_id = f"{session_id}:track:{track_id}" if track_id is not None else f"{session_id}:frame-{frame_number}:det-{idx}"
            object_record = {
                "class_name": item.get("class_name") or item.get("model_class_name") or "object",
                "track_id": track_id,
                "detection_id": det_id,
                "confidence": item.get("confidence"),
                "horizontal_position": item.get("horizontal_position") or "UNKNOWN",
                "depth_category": item.get("depth_category") or "UNKNOWN",
                "bounding_box": item.get("bounding_box"),
            }
            sample_objects.append(object_record)
            identity = (
                object_record["class_name"],
                str(object_record["track_id"]) if object_record["track_id"] is not None else
                f"{object_record['horizontal_position']}:{object_record['depth_category']}",
            )
            unique_objects[identity] = object_record

        review_token = uuid4().hex
        raw_frame = frame_result.get("review_frame_image")
        if raw_frame:
            _cache_review_frame(review_token, raw_frame)

        samples.append({
            "frame_number": frame_number,
            "timestamp_seconds": timestamp_seconds,
            "status": frame_status,
            "session_id": session_id,
            "review_frame_token": review_token,
            "annotated_image": frame_result.get("annotated_image"),
            "annotated_size": frame_result.get("annotated_size"),
            "objects": sample_objects,
            "detection_count": len(sample_objects),
            "scene_description": description if isinstance(description, str) else "",
            "warnings": list(frame_result.get("warnings", [])),
        })

    scenes = distinct_descriptions[:3]
    if len(distinct_descriptions) > 3:
        scenes.append(distinct_descriptions[-1])
    if scenes:
        summary = f"Across {result['frames_processed']} sampled video frames, " + " Later, ".join(scenes)
    else:
        names = sorted({item["class_name"] for item in unique_objects.values()})
        summary = (
            f"Across {result['frames_processed']} sampled video frames, BYAKU observed "
            + (", ".join(names) if names else "no recognizable objects")
            + "."
        )

    total_detections = sum(s.get("detection_count", 0) for s in samples)
    video_warnings = list(result.get("warnings", []))
    if result.get("final_status") == "partial" and not video_warnings:
        video_warnings.append("Some sampled frames or depth estimates were unavailable.")

    return {
        "type": "upload_result",
        "media_type": "video",
        "status": result.get("final_status", "success"),
        "session_id": session_id,
        "frames_read": result.get("frames_read", 0),
        "frames_processed": result.get("frames_processed", 0),
        "frames_skipped": result.get("frames_skipped", 0),
        "frames_failed": result.get("frames_failed", 0),
        "frame_stride": _UPLOAD_VIDEO_FRAME_STRIDE,
        "samples": samples,
        "objects": list(unique_objects.values()),
        "detection_count": total_detections,
        "scene_description": summary,
        "warnings": video_warnings,
    }


@app.post("/api/upload/image")
async def process_uploaded_image(upload: UploadFile) -> dict[str, Any]:
    suffix = _validate_upload_suffix(upload, _UPLOAD_IMAGE_SUFFIXES, "image")
    try:
        content = await _read_bounded_upload(upload, _UPLOAD_IMAGE_MAX_BYTES, "image")
    finally:
        await upload.close()
    if suffix == ".png":
        valid_signature = content.startswith(b"\x89PNG\r\n\x1a\n")
    else:
        valid_signature = content.startswith(b"\xff\xd8\xff")
    if not valid_signature:
        raise HTTPException(status_code=415, detail="The image content does not match its file extension.")
    frame = cv2.imdecode(np.frombuffer(content, dtype=np.uint8), cv2.IMREAD_COLOR)
    if frame is None:
        raise HTTPException(status_code=415, detail="The uploaded image is invalid or cannot be decoded.")
    if frame.shape[0] * frame.shape[1] > _UPLOAD_IMAGE_MAX_PIXELS:
        raise HTTPException(status_code=413, detail="The image dimensions exceed BYAKU's safe processing limit.")
    try:
        return await run_in_threadpool(_process_uploaded_image, frame)
    except HTTPException:
        raise
    except Exception as error:
        _logger.exception("Uploaded image processing failed")
        raise HTTPException(status_code=503, detail="BYAKU could not process this image right now. Please try again.") from error


@app.post("/api/upload/video")
async def process_uploaded_video(upload: UploadFile) -> dict[str, Any]:
    suffix = _validate_upload_suffix(upload, _UPLOAD_VIDEO_SUFFIXES, "video")
    try:
        with tempfile.TemporaryDirectory(prefix="byaku-upload-") as temporary_directory:
            video_path = await _store_video_upload(upload, suffix, Path(temporary_directory))
            return await run_in_threadpool(_process_uploaded_video, video_path)
    except HTTPException:
        raise
    except Exception as error:
        _logger.exception("Uploaded video processing failed")
        raise HTTPException(status_code=503, detail="BYAKU could not process this video right now. Please try again.") from error
    finally:
        await upload.close()


def _process_live_frame_unlocked(request: FrameRequest) -> dict[str, Any]:
    """Process a frame while the caller owns the shared frame-processing lock."""
    profiler = getattr(app.state, "frame_profiler", None)
    if not isinstance(profiler, FrameProfiler):
        profiler = None
    adapters_need_initialization = any(
        getattr(app.state, adapter_name, None) is None
        for adapter_name in ("detection_adapter", "depth_adapter", "orchestrator")
    )
    initialization_started = time.perf_counter()
    with profile_stage(profiler, "adapter_lookup_or_initialization"):
        detection_adapter, depth_adapter, orchestrator = _ensure_live_adapters()
    initialization_ms = (
        round((time.perf_counter() - initialization_started) * 1000.0, 1)
        if adapters_need_initialization
        else 0.0
    )
    _t0 = time.perf_counter()
    _validate_frame_identity(orchestrator, frame_id=request.frame_id, frame_number=request.frame_number)
    app.state.last_assigned_frame_number = max(
        getattr(app.state, "last_assigned_frame_number", 0),
        request.frame_number,
    )

    with profile_stage(profiler, "jpeg_base64_decode_and_opencv_decode"):
        frame = _decode_base64_jpeg(request.data)
    if frame.shape[0] != request.height or frame.shape[1] != request.width:
        raise HTTPException(
            status_code=400,
            detail=(
                "frame dimensions do not match the declared width/height; "
                f"decoded {frame.shape[1]}x{frame.shape[0]} but request specified {request.width}x{request.height}"
            ),
        )

    with profile_stage(profiler, "depth_anything_total"):
        if isinstance(depth_adapter, DepthAnythingV2Adapter):
            depth_result = depth_adapter.estimate_depth(frame, profiler=profiler)
        else:
            depth_result = depth_adapter.estimate_depth(frame)
    with profile_stage(profiler, "depth_result_validation"):
        depth_map, depth_warnings = _validated_depth_map(depth_result, frame_shape=frame.shape[:2])

    detection_state = _snapshot_detection_identity(detection_adapter)
    try:
        with profile_stage(profiler, "yolo_bytetrack_total"):
            detection_result = detection_adapter.process_frame(
                frame,
                frame_id=request.frame_id,
                frame_number=request.frame_number,
            )
    except Exception:
        _restore_detection_identity(detection_adapter, detection_state)
        raise
    if detection_result.get("status") not in {"success", "partial"}:
        _restore_detection_identity(detection_adapter, detection_state)
        raise HTTPException(
            status_code=400,
            detail=detection_result.get("warnings", ["detection processing failed"])[0],
        )

    correction_warnings: list[str] = []
    remembered_by_identity, corrections_to_mark = _apply_remembered_corrections(
        frame,
        detection_result.get("orchestrator_detections", []),
        warnings=correction_warnings,
    )
    with profile_stage(profiler, "pipeline_orchestration_call"):
        pipeline_result = orchestrator.process_frame(
            frame_id=request.frame_id,
            frame_number=request.frame_number,
            image_dimensions={"width": request.width, "height": request.height},
            detections=detection_result.get("orchestrator_detections", []),
            depth_map=depth_map,
            timestamp=request.timestamp,
            profiler=profiler,
        )
    if pipeline_result.get("status") in {"invalid_input", "error"}:
        _restore_detection_identity(detection_adapter, detection_state)
        raise HTTPException(
            status_code=400,
            detail=pipeline_result.get("warnings", ["frame processing failed"])[0],
        )
    applied_at = datetime.now(UTC).isoformat()
    for metadata in corrections_to_mark:
        correction_id = metadata.get("correction_id")
        if not isinstance(correction_id, str):
            continue
        try:
            marked = _feedback_store.mark_applied_to_future_inference(
                correction_id,
                applied_at=applied_at,
            )
        except (OSError, ValueError, json.JSONDecodeError) as error:
            correction_warnings.append(f"correction memory could not be marked as applied: {error}")
        else:
            metadata["applied"] = marked is not None
    _attach_remembered_correction_metadata(
        pipeline_result.get("objects"),
        remembered_by_identity,
    )
    with profile_stage(profiler, "response_payload_preparation"):
        response = _serialize_frame_result(pipeline_result, request=request, depth_map=depth_map)
    response["warnings"] = list(dict.fromkeys([*response["warnings"], *depth_warnings, *correction_warnings]))
    if depth_warnings:
        response["status"] = "partial"
    stage_status = _stage_status(
        detection_result=detection_result,
        depth_result=depth_result,
        pipeline_result=pipeline_result,
    )
    response["stage_status"] = stage_status
    response["initialization_ms"] = initialization_ms
    response["processing_ms"] = round((time.perf_counter() - _t0) * 1000.0, 1)
    response["timing_source"] = "server_perf_counter"
    review_frame_token = uuid4().hex
    response["review_frame_token"] = review_frame_token
    now_monotonic = time.monotonic()
    with _review_frame_cache_lock:
        _review_frame_cache[review_frame_token] = (request.data, now_monotonic)
        expired = [
            token for token, (_, captured_at) in _review_frame_cache.items()
            if now_monotonic - captured_at > 180.0
        ]
        for token in expired:
            _review_frame_cache.pop(token, None)
        while len(_review_frame_cache) > 64:
            _review_frame_cache.pop(next(iter(_review_frame_cache)))
    app.state.latest_frame_result = response
    app.state.latest_frame_preview_data_url = f"data:image/jpeg;base64,{request.data}"
    app.state.latest_stage_status = stage_status
    if profiler is not None:
        profiler.record("backend_initialization_total", initialization_ms)
        profiler.record("backend_processing_total", (time.perf_counter() - _t0) * 1000.0)
    return response


@app.post("/api/frame")
def process_live_frame(request: FrameRequest) -> dict[str, Any]:
    """Process one frame; report adapter/model setup separately from frame processing."""
    if not _frame_processing_lock.acquire(blocking=False):
        raise HTTPException(status_code=409, detail="frame processing is busy")
    try:
        return _process_live_frame_unlocked(request)
    finally:
        _frame_processing_lock.release()


def _process_live_query_unlocked(request: QueryRequest) -> dict[str, Any]:
    """Answer a query while the caller owns the shared processing lock."""
    _, _, orchestrator = _ensure_live_adapters()
    committed_at = getattr(orchestrator, "_latest_commit_wall_time", None)
    scene_age_seconds = (
        max(0.0, time.monotonic() - committed_at)
        if isinstance(committed_at, (int, float)) and not isinstance(committed_at, bool)
        else None
    )
    result = orchestrator.query(request.query, scene_age_seconds=scene_age_seconds)
    if result.get("status") == "not_ready":
        response = {
            "type": "query_result",
            "query": request.query,
            "frame_id": result.get("frame_id"),
            "frame_number": result.get("frame_number"),
            "status": "not_ready",
            "answer": None,
            "matched_objects": [],
            "evidence": [],
            "warnings": list(result.get("warnings", [])),
            "speech_response": result.get("speech_response"),
        }
    else:
        response = {
            "type": "query_result",
            "query": request.query,
            "frame_id": result.get("frame_id"),
            "frame_number": result.get("frame_number"),
            "status": result.get("status", "unsupported"),
            "answer": result.get("query_result", {}).get("answer"),
            "matched_objects": list(result.get("query_result", {}).get("matched_objects", [])),
            "evidence": list(result.get("query_result", {}).get("evidence", [])),
            "warnings": list(result.get("warnings", [])),
            "speech_response": result.get("speech_response"),
        }
    app.state.latest_query_response = response
    stage_status = _latest_stage_status()
    query_result = result.get("query_result")
    reasoning_ran = (
        result.get("status") not in {"not_ready", "error"}
        and isinstance(query_result, Mapping)
    )
    stage_status["reasoning"] = "ready" if reasoning_ran else "unavailable"
    speech = response.get("speech_response")
    stage_status["narration"] = "ready" if isinstance(speech, Mapping) and speech.get("spoken_text") else "unavailable"
    app.state.latest_stage_status = stage_status
    return response


@app.post("/api/query")
def process_live_query(request: QueryRequest) -> dict[str, Any]:
    """Answer a query against the latest committed scene."""
    if not _frame_processing_lock.acquire(blocking=False):
        raise HTTPException(status_code=409, detail="scene processing is busy")
    try:
        return _process_live_query_unlocked(request)
    finally:
        _frame_processing_lock.release()


@app.get("/api/feedback")
def list_feedback(limit: int = 100) -> dict[str, Any]:
    if not 1 <= limit <= 500:
        raise HTTPException(status_code=422, detail="limit must be between 1 and 500")
    try:
        return {"schema_version": FeedbackStore.SCHEMA_VERSION, "feedback": _feedback_store.list(limit=limit)}
    except (OSError, ValueError, json.JSONDecodeError) as error:
        raise HTTPException(status_code=500, detail=f"feedback history could not be read: {error}") from error


@app.post("/api/feedback", status_code=201)
def save_feedback(request: FeedbackRequest) -> dict[str, Any]:
    corrected_name = request.human_corrected_name
    if corrected_name is None and request.correction_type == "class_correction" and isinstance(request.corrected_value, str):
        corrected_name = " ".join(request.corrected_value.split())
    if request.correction_type == "class_correction" and not corrected_name:
        raise HTTPException(status_code=422, detail="human_corrected_name is required for a class correction")
    if request.correction_type == "confirm" and request.review_status not in {"confirmed", "accepted"}:
        raise HTTPException(status_code=422, detail="confirmed detections must use review_status 'confirmed'")
    try:
        existing = _feedback_store.get_by_submission_id(request.submission_id)
        if existing is not None:
            return existing
    except (OSError, ValueError, json.JSONDecodeError) as error:
        raise HTTPException(status_code=500, detail=f"feedback could not be read: {error}") from error

    now = datetime.now(UTC).isoformat()
    record = request.model_dump()
    record["human_corrected_name"] = corrected_name
    original = record["original_prediction"]
    feedback_id = uuid4().hex
    source_val = request.source or "live"
    record.update({
        "schema_version": FeedbackStore.SCHEMA_VERSION,
        "feedback_id": feedback_id,
        "correction_id": feedback_id,
        "created_at": now,
        "timestamp": now,
        "source": source_val,
        "session_id": request.session_id,
        "frame_number": request.frame_number,
        "track_id": request.track_id,
        "detection_id": request.detection_id,
        "original_predicted_class": original["class_name"],
        "original_confidence": original.get("confidence"),
        "image_dimensions": record.pop("image_dimensions", None),
        "frame_reference": {
            "session_id": request.session_id,
            "frame_id": request.frame_id,
            "frame_number": request.frame_number,
            "review_frame_token": request.review_frame_token,
            "source": source_val,
            "detection_id": request.detection_id,
        },
        "human_corrected_name": corrected_name,
        "applied_to_future_inference": False,
        "applied_at": None,
        "image_reference": None,
    })
    record["visual_fingerprint"] = None
    frame_data: str | None = None
    if request.review_frame_token:
        with _review_frame_cache_lock:
            cached = _review_frame_cache.get(request.review_frame_token)
            if cached is not None and time.monotonic() - cached[1] <= 180.0:
                frame_data = cached[0]
    frame_snapshot = _decode_base64_jpeg(frame_data) if frame_data is not None else None
    if frame_snapshot is not None:
        dimensions = {"width": int(frame_snapshot.shape[1]), "height": int(frame_snapshot.shape[0])}
        record["image_dimensions"] = dimensions
        if request.correction_type == "class_correction":
            record["visual_fingerprint"] = _visual_fingerprint(
                frame_snapshot,
                original.get("bounding_box"),
            )
    record["corrected_value"] = (
        corrected_name
        if request.correction_type == "class_correction"
        else request.corrected_value
    )
    try:
        return _feedback_store.create(record)
    except (OSError, ValueError, json.JSONDecodeError) as error:
        raise HTTPException(status_code=500, detail=f"feedback could not be saved: {error}") from error


@app.post("/api/feedback/{feedback_id}/undo")
def undo_feedback(feedback_id: str) -> dict[str, Any]:
    try:
        record = _feedback_store.undo(
            feedback_id,
            undone_at=datetime.now(UTC).isoformat(),
        )
    except (OSError, ValueError, json.JSONDecodeError) as error:
        raise HTTPException(status_code=500, detail=f"feedback could not be undone: {error}") from error
    if record is None:
        raise HTTPException(status_code=404, detail="feedback record was not found")
    return record


@app.get("/api/status")
def api_status() -> dict[str, Any]:
    """Report connectivity separately from last_frame_available.

    processing_ms excludes adapter/model setup, initialization_ms measures that setup,
    and timing_source identifies the server-side clock used for both.
    """
    camera_connected = _browser_camera_connected() or _python_camera_running()
    stage_status = _latest_stage_status()
    stage_status["camera"] = "connected" if camera_connected else "disconnected"
    frame_result = getattr(app.state, "latest_frame_result", None)
    latest_processing_ms: float | None = None
    latest_initialization_ms: float | None = None
    timing_source: str | None = None
    if isinstance(frame_result, Mapping):
        frame_result = dict(frame_result)
        frame_result["stage_status"] = stage_status
        raw_ms = frame_result.get("processing_ms")
        if isinstance(raw_ms, (int, float)) and not isinstance(raw_ms, bool) and raw_ms >= 0:
            latest_processing_ms = float(raw_ms)
        raw_initialization_ms = frame_result.get("initialization_ms")
        if (
            isinstance(raw_initialization_ms, (int, float))
            and not isinstance(raw_initialization_ms, bool)
            and raw_initialization_ms >= 0
        ):
            latest_initialization_ms = float(raw_initialization_ms)
        raw_timing_source = frame_result.get("timing_source")
        if isinstance(raw_timing_source, str):
            timing_source = raw_timing_source

    query_response = getattr(app.state, "latest_query_response", None)
    speech = query_response.get("speech_response") if isinstance(query_response, Mapping) else None
    response: dict[str, Any] = {
        "backend": "connected",
        "camera": stage_status["camera"],
        "stage_status": stage_status,
        "last_frame": frame_result,
        "last_frame_available": frame_result is not None,
        "last_query": query_response.get("query") if isinstance(query_response, Mapping) else None,
        "last_reasoning": dashboard_reasoning(query_response),
        "last_narration": speech.get("spoken_text") if isinstance(speech, Mapping) else None,
    }
    frame_preview = getattr(app.state, "latest_frame_preview_data_url", None)
    if isinstance(frame_preview, str):
        response["last_frame_preview"] = frame_preview
    if latest_processing_ms is not None:
        response["processing_ms"] = latest_processing_ms
    if latest_initialization_ms is not None:
        response["initialization_ms"] = latest_initialization_ms
    if timing_source is not None:
        response["timing_source"] = timing_source
    return response


@app.post("/api/camera/start", response_model=None)
def start_camera(request: CameraStartRequest = CameraStartRequest()) -> dict[str, Any] | JSONResponse:
    if not _camera_operation_lock.acquire(blocking=False):
        return _camera_error(
            status_code=409,
            code="camera_busy",
            message="a camera operation is already in progress",
            recoverable=True,
        )
    try:
        if _browser_camera_connected():
            return _camera_error(
                status_code=409,
                code="camera_busy",
                message="a browser camera session is active",
                recoverable=True,
            )
        producer = getattr(app.state, "camera_producer", None)
        if producer is not None and producer.is_running and producer.source != request.source:
            return _camera_error(
                status_code=409,
                code="camera_already_running",
                message=f"camera source {producer.source!r} is already running",
                recoverable=True,
            )
        if producer is None or producer.source != request.source:
            producer = _new_camera_producer(request.source)
            app.state.camera_producer = producer

        result = producer.start()
        if result.get("status") not in {"started", "already_running"}:
            unavailable = result.get("status") == "unavailable"
            return _camera_error(
                status_code=503,
                code="camera_unavailable" if unavailable else "camera_open_failed",
                message=result.get("message", "camera could not be started"),
                recoverable=True,
            )
        return {"type": "camera_status", **result}
    except Exception as error:
        producer = getattr(app.state, "camera_producer", None)
        if producer is not None:
            producer.stop()
        return _camera_error(
            status_code=503,
            code="camera_open_failed",
            message=f"camera start failed: {type(error).__name__}: {error}",
            recoverable=True,
        )
    finally:
        _camera_operation_lock.release()


def _websocket_status() -> dict[str, Any]:
    stage_status = _latest_stage_status()
    stage_status["camera"] = "connected" if (_browser_camera_connected() or _python_camera_running()) else "disconnected"
    last_frame = getattr(app.state, "latest_frame_result", None)
    return {
        "type": "status",
        "backend": "connected",
        "camera": stage_status["camera"],
        "last_frame_available": last_frame is not None,
        "stage_status": stage_status,
    }


@app.websocket("/ws")
async def websocket_protocol(websocket: WebSocket) -> None:
    await websocket.accept()
    session_id = uuid4().hex
    with _websocket_sessions_lock:
        _websocket_sessions()[session_id] = {"camera_active": False, "explicitly_stopped": False}
    await websocket.send_json(_websocket_status())

    try:
        while True:
            receive_started = time.perf_counter()
            incoming = await websocket.receive()
            receive_elapsed_ms = (time.perf_counter() - receive_started) * 1000.0
            if incoming["type"] == "websocket.disconnect":
                break
            text = incoming.get("text")
            if not isinstance(text, str):
                await websocket.send_json(
                    _websocket_error("invalid_message", "expected a text JSON message", stage="protocol")
                )
                continue
            json_parse_started = time.perf_counter()
            try:
                message = json.loads(text)
            except (TypeError, ValueError):
                await websocket.send_json(
                    _websocket_error("invalid_json", "message must contain valid JSON", stage="protocol")
                )
                continue
            json_parse_elapsed_ms = (time.perf_counter() - json_parse_started) * 1000.0
            if not isinstance(message, Mapping) or not isinstance(message.get("type"), str):
                await websocket.send_json(
                    _websocket_error("invalid_message", "message must be a JSON object with a type", stage="protocol")
                )
                continue

            message_type = message["type"]
            if message_type == "ping":
                pong: dict[str, Any] = {"type": "pong"}
                if "ts" in message:
                    pong["ts"] = message["ts"]
                await websocket.send_json(pong)
                continue

            if message_type in {"start_camera", "stop_camera"}:
                if not _camera_operation_lock.acquire(blocking=False):
                    await websocket.send_json(
                        _websocket_error("camera_busy", "a camera operation is already in progress")
                    )
                    continue
                try:
                    if message_type == "start_camera" and _python_camera_running():
                        await websocket.send_json(
                            _websocket_error(
                                "camera_busy",
                                "the Python camera producer is active; browser camera processing was not started",
                            )
                        )
                        continue
                    with _websocket_sessions_lock:
                        session = _websocket_sessions().get(session_id)
                        if session is None:
                            break
                        if message_type == "start_camera":
                            session["camera_active"] = True
                            session["explicitly_stopped"] = False
                        else:
                            session["camera_active"] = False
                            session["explicitly_stopped"] = True
                    await websocket.send_json(_websocket_status())
                finally:
                    _camera_operation_lock.release()
                continue

            if message_type == "frame":
                profiler = getattr(app.state, "frame_profiler", None)
                if not isinstance(profiler, FrameProfiler):
                    profiler = None
                if profiler is not None:
                    profiler.record("websocket_receive", receive_elapsed_ms)
                    profiler.record("websocket_json_parse", json_parse_elapsed_ms)
                client_frame_id = message.get("frame_id")
                if (
                    isinstance(client_frame_id, bool)
                    or not isinstance(client_frame_id, int)
                    or client_frame_id <= 0
                ):
                    await websocket.send_json(
                        _websocket_error(
                            "invalid_frame_id",
                            "frame_id must be a positive integer",
                            frame_id=client_frame_id if isinstance(client_frame_id, int) else None,
                        )
                    )
                    continue
                width = message.get("width")
                height = message.get("height")
                data = message.get("data")
                if (
                    isinstance(width, bool)
                    or not isinstance(width, int)
                    or width <= 0
                    or isinstance(height, bool)
                    or not isinstance(height, int)
                    or height <= 0
                    or not isinstance(data, str)
                    or not data.strip()
                ):
                    await websocket.send_json(
                        _websocket_error(
                            "invalid_frame",
                            "frame requires positive integer width/height and non-empty base64 JPEG data",
                            frame_id=client_frame_id,
                        )
                    )
                    continue

                session_error: tuple[str, str] | None = None
                session_active = False
                with _websocket_sessions_lock:
                    session = _websocket_sessions().get(session_id)
                    if session is None:
                        session_error = ("session_closed", "WebSocket session is no longer active")
                    elif session.get("explicitly_stopped"):
                        session_error = ("camera_not_started", "send start_camera before submitting more frames")
                    elif client_frame_id <= session.get("last_client_frame_id", 0):
                        session_error = ("duplicate_frame_id", "frame_id must increase within the WebSocket session")
                    else:
                        session_active = bool(session.get("camera_active"))
                if session_error is not None:
                    code, detail = session_error
                    await websocket.send_json(
                        _websocket_error(code, detail, frame_id=client_frame_id)
                    )
                    if code == "session_closed":
                        break
                    continue

                if not session_active:
                    if not _camera_operation_lock.acquire(blocking=False):
                        await websocket.send_json(
                            _websocket_error("camera_busy", "a camera operation is already in progress", frame_id=client_frame_id)
                        )
                        continue
                    try:
                        if _python_camera_running():
                            await websocket.send_json(
                                _websocket_error(
                                    "camera_busy",
                                    "the Python camera producer is active; browser frame processing was not started",
                                    frame_id=client_frame_id,
                                )
                            )
                            continue
                        with _websocket_sessions_lock:
                            session = _websocket_sessions().get(session_id)
                            if session is not None:
                                session["camera_active"] = True
                    finally:
                        _camera_operation_lock.release()

                if not _frame_processing_lock.acquire(blocking=False):
                    await websocket.send_json(
                        _websocket_error(
                            "frame_processing_busy",
                            "another frame or query is being processed",
                            frame_id=client_frame_id,
                        )
                    )
                    continue
                session_closed = False
                session_stopped = False
                with _websocket_sessions_lock:
                    session = _websocket_sessions().get(session_id)
                    if session is None:
                        session_closed = True
                    elif session.get("explicitly_stopped"):
                        session_stopped = True
                    else:
                        session["camera_active"] = True
                        session["last_client_frame_id"] = client_frame_id
                if session_closed or session_stopped:
                    _frame_processing_lock.release()
                    if session_stopped:
                        await websocket.send_json(
                            _websocket_error(
                                "camera_not_started",
                                "send start_camera before submitting more frames",
                                frame_id=client_frame_id,
                            )
                        )
                    else:
                        break
                    continue

                server_frame_number = _next_websocket_frame_number()
                internal_frame_id = f"browser:{session_id}:{client_frame_id}"
                try:
                    with profile_stage(profiler, "frame_request_validation"):
                        frame_request = FrameRequest(
                            frame_id=internal_frame_id,
                            frame_number=server_frame_number,
                            width=width,
                            height=height,
                            data=data,
                            timestamp=time.time(),
                        )
                    result = await run_in_threadpool(_process_live_frame_unlocked, frame_request)
                    with profile_stage(profiler, "browser_response_adaptation"):
                        result = adapt_browser_frame_result(result, browser_frame_id=client_frame_id)
                    result["frame_number"] = server_frame_number
                    app.state.latest_frame_result = result
                    await _send_profiled_frame_result(websocket, result, profiler)
                except HTTPException as error:
                    detail = error.detail if isinstance(error.detail, str) else "frame processing failed"
                    code = "invalid_frame" if error.status_code == 400 and (
                        detail.startswith("frame data") or detail.startswith("frame dimensions")
                    ) else "frame_processing_failed"
                    await websocket.send_json(
                        _websocket_error(code, detail, recoverable=True, frame_id=client_frame_id)
                    )
                except Exception as error:
                    await websocket.send_json(
                        _websocket_error(
                            "frame_processing_failed",
                            f"frame processing failed: {type(error).__name__}: {error}",
                            recoverable=True,
                            frame_id=client_frame_id,
                        )
                    )
                finally:
                    _frame_processing_lock.release()
                continue

            if message_type == "query":
                query_text = message.get("text")
                if not isinstance(query_text, str) or not query_text.strip():
                    await websocket.send_json(
                        _websocket_error(
                            "invalid_query",
                            "query message requires non-empty text",
                            stage="query",
                        )
                    )
                    continue
                # Wait briefly for any active frame to finish before rejecting the query.
                if not _frame_processing_lock.acquire(timeout=5.0):
                    await websocket.send_json(
                        _websocket_error(
                            "processing_busy",
                            "a frame or query is being processed; please try again",
                            stage="query",
                        )
                    )
                    continue
                try:
                    result = await run_in_threadpool(
                        _process_live_query_unlocked,
                        QueryRequest(query=query_text.strip()),
                    )
                    response = adapt_browser_query_result(result)
                    await websocket.send_json(response)
                except Exception as error:
                    await websocket.send_json(
                        _websocket_error(
                            "query_failed",
                            f"query processing failed: {type(error).__name__}: {error}",
                            stage="query",
                        )
                    )
                finally:
                    _frame_processing_lock.release()
                continue

            await websocket.send_json(
                _websocket_error(
                    "unsupported_message",
                    f"unsupported message type: {message_type}",
                    stage="protocol",
                )
            )
    except WebSocketDisconnect:
        pass
    finally:
        with _websocket_sessions_lock:
            _websocket_sessions().pop(session_id, None)


@app.post("/api/camera/capture", response_model=None)
def capture_camera_frame() -> dict[str, Any] | JSONResponse:
    if not _camera_operation_lock.acquire(blocking=False):
        return _camera_error(
            status_code=409,
            code="camera_busy",
            message="a camera operation is already in progress",
            recoverable=True,
        )
    try:
        producer = getattr(app.state, "camera_producer", None)
        if producer is None or not producer.is_running:
            return _camera_error(
                status_code=409,
                code="camera_not_running",
                message="camera has not been started",
                recoverable=True,
            )
        try:
            result = producer.capture_once()
        except Exception as error:
            producer.stop()
            return _camera_error(
                status_code=502,
                code="frame_processing_failed",
                message=f"camera frame processing failed: {type(error).__name__}: {error}",
                recoverable=True,
            )

        status = result.get("status")
        if status == "submitted":
            frame_result = result["frame_result"]
            return {
                "type": "camera_capture_result",
                "status": frame_result["status"],
                "frame_id": result["frame_id"],
                "frame_number": result["frame_number"],
                "frame_result": frame_result,
            }
        if status == "read_failed":
            return _camera_error(
                status_code=503,
                code="camera_read_failed",
                message=result.get("message", "camera did not provide a frame"),
                recoverable=bool(result.get("recoverable")),
            )
        if status == "capture_error":
            return _camera_error(
                status_code=503,
                code="camera_read_failed",
                message=result.get("message", "camera read failed"),
                recoverable=True,
            )
        if status == "encode_failed":
            return _camera_error(
                status_code=503,
                code="camera_encode_failed",
                message=result.get("message", "camera capture failed"),
                recoverable=True,
            )
        if status == "submission_failed":
            return _camera_error(
                status_code=502,
                code="frame_processing_failed",
                message=result.get("message", "frame processing failed"),
                recoverable=True,
            )
        return _camera_error(
            status_code=409,
            code="camera_not_running",
            message=result.get("message", "camera is not running"),
            recoverable=True,
        )
    finally:
        _camera_operation_lock.release()


@app.post("/api/camera/stop", response_model=None)
def stop_camera() -> dict[str, Any] | JSONResponse:
    if not _camera_operation_lock.acquire(blocking=False):
        return _camera_error(
            status_code=409,
            code="camera_busy",
            message="a camera operation is already in progress",
            recoverable=True,
        )
    try:
        producer = getattr(app.state, "camera_producer", None)
        if producer is None:
            return {"type": "camera_status", "status": "stopped", "already_stopped": True}
        result = producer.stop()
        return {"type": "camera_status", **result}
    except Exception as error:
        return _camera_error(
            status_code=500,
            code="camera_stop_failed",
            message=f"camera stop failed: {type(error).__name__}: {error}",
            recoverable=True,
        )
    finally:
        _camera_operation_lock.release()