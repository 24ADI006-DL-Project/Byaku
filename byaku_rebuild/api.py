import base64
from contextlib import asynccontextmanager
from collections.abc import Mapping
import json
from threading import Lock
import time
from typing import Any
from uuid import uuid4

import cv2
import numpy as np
from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator
from starlette.concurrency import run_in_threadpool
from starlette.exceptions import HTTPException as StarletteHTTPException

from .detection_tracking import DEFAULT_WEIGHTS, DetectionTrackingAdapter
from .depth_integration import DepthAnythingV2Adapter, MODEL_NAME
from .browser_protocol_adapter import (
    adapt_browser_frame_result,
    adapt_browser_query_result,
    dashboard_reasoning,
)
from .frame_profiler import FrameProfiler, profile_stage
from .pipeline_orchestrator import PipelineOrchestrator

_camera_operation_lock = Lock()
_frame_processing_lock = Lock()
_websocket_sessions_lock = Lock()


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
    return {
        "camera": "connected" if active_camera else "disconnected",
        "detection": _component_status(detection_status),
        "tracking": _component_status(detection_status),
        "depth": _component_status(depth_status),
        "spatial": _component_status(pipeline_status),
        "scene_graph": "ready" if graph_available else "unavailable",
        "reasoning": "ready" if getattr(app.state, "latest_query_response", None) else "unavailable",
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
        raw_errors = exception.errors(include_url=False)
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
    depth_pixels = np.clip(preview_depth * 255.0, 0, 255).astype(np.uint8)
    colorized_depth = cv2.applyColorMap(depth_pixels, cv2.COLORMAP_TURBO)
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
        "description": "Live frame processing, scene queries, and camera capture API.",
    }


def _process_live_frame_unlocked(request: FrameRequest) -> dict[str, Any]:
    """Process a frame while the caller owns the shared frame-processing lock."""
    _t0 = time.perf_counter()
    profiler = getattr(app.state, "frame_profiler", None)
    if not isinstance(profiler, FrameProfiler):
        profiler = None
    with profile_stage(profiler, "adapter_lookup_or_initialization"):
        detection_adapter, depth_adapter, orchestrator = _ensure_live_adapters()
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
    with profile_stage(profiler, "response_payload_preparation"):
        response = _serialize_frame_result(pipeline_result, request=request, depth_map=depth_map)
    response["warnings"] = list(dict.fromkeys([*response["warnings"], *depth_warnings]))
    if depth_warnings:
        response["status"] = "partial"
    stage_status = _stage_status(
        detection_result=detection_result,
        depth_result=depth_result,
        pipeline_result=pipeline_result,
    )
    stage_status["reasoning"] = "unavailable"
    response["stage_status"] = stage_status
    # Measure actual wall-clock processing time and attach it for the browser latency counter.
    response["processing_ms"] = round((time.perf_counter() - _t0) * 1000.0, 1)
    app.state.latest_frame_result = response
    app.state.latest_stage_status = stage_status
    if profiler is not None:
        profiler.record("backend_processing_total", (time.perf_counter() - _t0) * 1000.0)
    return response


@app.post("/api/frame")
def process_live_frame(request: FrameRequest) -> dict[str, Any]:
    """Accept one validated frame and process it through the existing modules."""
    if not _frame_processing_lock.acquire(blocking=False):
        raise HTTPException(status_code=409, detail="frame processing is busy")
    try:
        return _process_live_frame_unlocked(request)
    finally:
        _frame_processing_lock.release()


def _process_live_query_unlocked(request: QueryRequest) -> dict[str, Any]:
    """Answer a query while the caller owns the shared processing lock."""
    _, _, orchestrator = _ensure_live_adapters()
    result = orchestrator.query(request.query)
    if result.get("status") == "not_ready":
        response = {
            "type": "query_result",
            "query": request.query,
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
            "status": result.get("status", "unsupported"),
            "answer": result.get("query_result", {}).get("answer"),
            "matched_objects": list(result.get("query_result", {}).get("matched_objects", [])),
            "evidence": list(result.get("query_result", {}).get("evidence", [])),
            "warnings": list(result.get("warnings", [])),
            "speech_response": result.get("speech_response"),
        }
    app.state.latest_query_response = response
    stage_status = _latest_stage_status()
    stage_status["reasoning"] = "ready" if response.get("status") != "not_ready" else "unavailable"
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


@app.get("/api/status")
def api_status() -> dict[str, Any]:
    camera_connected = _browser_camera_connected() or _python_camera_running()
    stage_status = _latest_stage_status()
    stage_status["camera"] = "connected" if camera_connected else "disconnected"
    frame_result = getattr(app.state, "latest_frame_result", None)
    latest_processing_ms: float | None = None
    if isinstance(frame_result, Mapping):
        frame_result = dict(frame_result)
        frame_result["stage_status"] = stage_status
        raw_ms = frame_result.get("processing_ms")
        if isinstance(raw_ms, (int, float)) and not isinstance(raw_ms, bool) and raw_ms >= 0:
            latest_processing_ms = float(raw_ms)

    query_response = getattr(app.state, "latest_query_response", None)
    speech = query_response.get("speech_response") if isinstance(query_response, Mapping) else None
    response: dict[str, Any] = {
        "backend": "connected",
        "camera": stage_status["camera"],
        "stage_status": stage_status,
        "last_frame": frame_result,
        "last_query": query_response.get("query") if isinstance(query_response, Mapping) else None,
        "last_reasoning": dashboard_reasoning(query_response),
        "last_narration": speech.get("spoken_text") if isinstance(speech, Mapping) else None,
    }
    if latest_processing_ms is not None:
        response["processing_ms"] = latest_processing_ms
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
    return {
        "type": "status",
        "backend": "connected",
        "camera": stage_status["camera"],
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