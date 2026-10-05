from __future__ import annotations

"""FastAPI backend for the BYAKU computer-vision assistant.

Canonical websocket message contract (backward compatible with existing runtime):

Browser -> backend:
  {"type": "start_camera"}
  {"type": "stop_camera"}
  {"type": "frame", "frame_id": 1, "width": 640, "height": 480, "data": "<base64 jpeg>"}
  {"type": "query", "text": "Where is my bottle?"}

Backend -> browser:
  {"type": "status", "stage_status": {...}}
  {"type": "frame_result", "frame_id": 1, "frame_size": {"width": 640, "height": 480}, "detections": [...], "tracks": [...], "objects": [...], "events": [...], "status": "Ready"}
  {"type": "query_result", "query": "Where is my bottle?", "query_result": {...}, "narration": "...", "audio": "<base64 wav>"}
  {"type": "error", "stage": "camera", "code": "invalid_frame", "message": "...", "recoverable": true}
"""

import base64
import binascii
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.concurrency import run_in_threadpool

from app import VisionAssistantApp


MAX_FRAME_BYTES = 4 * 1024 * 1024
MAX_FRAME_DIMENSION = 1920


def _error_payload(stage: str, code: str, message: str, recoverable: bool = True) -> dict[str, Any]:
    return {
        "type": "error",
        "stage": stage,
        "code": code,
        "message": message,
        "recoverable": recoverable,
    }


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def _component_status(assistant: VisionAssistantApp, name: str, component: Any) -> str:
    if component is not None:
        return "running"
    return "error" if name in assistant.component_errors else "unavailable"


def _encode_audio(audio_path: str | None) -> str | None:
    if not audio_path:
        return None
    path = Path(audio_path)
    try:
        return base64.b64encode(path.read_bytes()).decode("ascii")
    except OSError:
        return None
    finally:
        try:
            path.unlink(missing_ok=True)
        except OSError:
            pass


def _depth_preview(depth_map: Any) -> str | None:
    if not isinstance(depth_map, np.ndarray) or depth_map.ndim != 2 or depth_map.size == 0:
        return None
    finite = np.isfinite(depth_map)
    if not finite.any():
        return None
    low = float(depth_map[finite].min())
    high = float(depth_map[finite].max())
    normalized = np.zeros(depth_map.shape, dtype=np.uint8)
    if high > low:
        normalized[finite] = np.clip((depth_map[finite] - low) * (255 / (high - low)), 0, 255).astype(np.uint8)
    preview = cv2.resize(normalized, (192, 108), interpolation=cv2.INTER_AREA)
    colored = cv2.applyColorMap(preview, cv2.COLORMAP_TURBO)
    ok, encoded = cv2.imencode(".jpg", colored, [cv2.IMWRITE_JPEG_QUALITY, 75])
    if not ok:
        return None
    return base64.b64encode(encoded.tobytes()).decode("ascii")


def create_api_app() -> FastAPI:
    state: dict[str, Any] = {
        "assistant": None,
        "camera_connected": False,
        "frames_processed": 0,
        "frame_times": [],
        "last_processing_ms": None,
        "last_query": None,
        "last_reasoning": None,
        "last_narration": None,
        "last_tts": "idle",
        "last_frame_status": "waiting",
        "last_frame": None,
        "first_scene_narrated": False,
    }

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        state["assistant"] = await run_in_threadpool(VisionAssistantApp, speak_events=False)
        try:
            yield
        finally:
            assistant = state["assistant"]
            if assistant is not None:
                await run_in_threadpool(assistant.close)

    api = FastAPI(title="BYAKU Python AI Backend", lifespan=lifespan)

    def get_assistant() -> VisionAssistantApp:
        assistant = state["assistant"]
        if assistant is None:
            raise HTTPException(status_code=503, detail="Python AI backend is starting")
        return assistant

    def build_status(include_last_frame: bool = True) -> dict[str, Any]:
        assistant = state["assistant"]
        if assistant is None:
            return {"backend": "starting", "camera": "disconnected"}
        current_time = time.monotonic()
        recent_times = [stamp for stamp in state["frame_times"] if current_time - stamp <= 5]
        state["frame_times"] = recent_times
        status = {
            "backend": "connected",
            "camera": "connected" if state["camera_connected"] else "disconnected",
            "detection": _component_status(assistant, "detector", assistant.detector),
            "tracking": _component_status(assistant, "tracker", assistant.tracker),
            "depth": _component_status(assistant, "depth_estimator", assistant.depth_estimator),
            "spatial": "running" if assistant.spatial_processor else "unavailable",
            "scene_graph": "running" if assistant.scene_memory and assistant.scene_memory.scene_graph else "unavailable",
            "reasoning": "ready" if assistant.query_engine else "unavailable",
            "narration": "ready" if assistant.narrator else "unavailable",
            "tts": "ready" if getattr(assistant.tts, "engine", None) is not None else "unavailable",
            "frames_processed": state["frames_processed"],
            "processing_fps": len(recent_times) / 5,
            "processing_ms": state["last_processing_ms"],
            "errors": dict(assistant.component_errors),
            "last_frame_status": state["last_frame_status"],
            "last_query": state["last_query"],
            "last_reasoning": state["last_reasoning"],
            "last_narration": state["last_narration"],
            "last_tts": state["last_tts"],
        }
        if include_last_frame:
            status["last_frame"] = state["last_frame"]
        return status

    def decode_frame(data: Any) -> np.ndarray:
        if not isinstance(data, str) or not data:
            raise ValueError("Frame data is required")
        if data.startswith("data:"):
            _, separator, data = data.partition(",")
            if not separator:
                raise ValueError("Frame data URL is invalid")
        try:
            encoded = base64.b64decode(data, validate=True)
        except (binascii.Error, ValueError) as exc:
            raise ValueError("Frame data must be base64-encoded JPEG") from exc
        if not encoded or len(encoded) > MAX_FRAME_BYTES:
            raise ValueError("Frame size is empty or exceeds 4 MB")
        frame = cv2.imdecode(np.frombuffer(encoded, dtype=np.uint8), cv2.IMREAD_COLOR)
        if frame is None or frame.size == 0:
            raise ValueError("Frame could not be decoded as an image")
        height, width = frame.shape[:2]
        if width > MAX_FRAME_DIMENSION or height > MAX_FRAME_DIMENSION:
            raise ValueError("Frame dimensions exceed 1920 pixels")
        return frame

    async def process_frame_message(payload: dict[str, Any]) -> dict[str, Any]:
        assistant = get_assistant()
        try:
            frame = decode_frame(payload.get("data"))
        except ValueError as exc:
            return _error_payload("camera", "invalid_frame", str(exc), recoverable=True)

        started = time.perf_counter()
        result = await run_in_threadpool(assistant.process_frame, frame)
        processing_ms = (time.perf_counter() - started) * 1000
        now = time.monotonic()
        state["frames_processed"] += 1
        state["frame_times"].append(now)
        state["last_processing_ms"] = round(processing_ms, 1)
        state["last_frame_status"] = result.get("status", "unknown")

        events = result.get("events", [])
        scene_narration = None
        if not state["first_scene_narrated"] and result.get("objects"):
            grouped: dict[str, list[dict[str, Any]]] = {}
            for item in result["objects"]:
                class_name = str(item.get("class_name", "")).strip()
                if class_name:
                    grouped.setdefault(class_name, []).append(item)
            scene_narration = " ".join(
                assistant.narrator.narrate({
                    "intent": "existence",
                    "found": True,
                    "target": class_name,
                    "objects": objects,
                    "status": "found",
                })
                for class_name, objects in list(grouped.items())[:3]
            ) or None
            state["first_scene_narrated"] = scene_narration is not None
        alert_audio = None
        audio_text = scene_narration
        if not audio_text:
            important_events = [event for event in events if event.get("message_required")]
            if important_events:
                audio_text = important_events[0].get("narration", "")
        if audio_text:
            alert_audio = await run_in_threadpool(assistant._synthesize_audio, audio_text)
            state["last_tts"] = "ready" if alert_audio else "unavailable"
        depth_map = result.get("depth_map")
        depth_preview = _depth_preview(depth_map)
        stage_status = build_status(include_last_frame=False)
        errors = result.get("status")
        if errors and errors != "Ready":
            stage_status["last_frame_status"] = errors
        response = {
            "type": "frame_result",
            "frame_id": payload.get("frame_id"),
            "frame_size": {"width": int(frame.shape[1]), "height": int(frame.shape[0])},
            "detections": _json_safe(result.get("detections", [])),
            "tracks": _json_safe(result.get("tracks", [])),
            "objects": _json_safe(result.get("objects", [])),
            "relationships": _json_safe(result.get("scene", {}).get("relationships", [])),
            "events": _json_safe(events),
            "depth_preview": depth_preview,
            "depth_available": depth_map is not None,
            "scene_narration": scene_narration,
            "scene_audio": _encode_audio(alert_audio) if scene_narration else None,
            "processing_ms": state["last_processing_ms"],
            "status": errors or "Ready",
            "stage_status": stage_status,
            "alert_audio": _encode_audio(alert_audio) if not scene_narration else None,
        }
        state["last_frame"] = {key: value for key, value in response.items() if key not in {"scene_audio", "alert_audio"}}
        return response

    async def process_query(payload: dict[str, Any]) -> dict[str, Any]:
        assistant = get_assistant()
        query = payload.get("text")
        if not isinstance(query, str) or not query.strip():
            return _error_payload("query", "empty_query", "A non-empty query is required", recoverable=True)
        if len(query) > 500:
            return _error_payload("query", "query_too_long", "Query exceeds 500 characters", recoverable=True)

        answer = await run_in_threadpool(assistant.answer_query, query.strip())
        audio = _encode_audio(answer.pop("audio", None))
        state["last_query"] = query.strip()
        state["last_reasoning"] = _json_safe(answer.get("query_result"))
        state["last_narration"] = answer.get("narration")
        state["last_tts"] = "ready" if audio else "unavailable"
        return {
            "type": "query_result",
            "query": query.strip(),
            **_json_safe(answer),
            "audio": audio,
            "stage_status": build_status(),
        }

    @api.get("/health")
    async def health() -> dict[str, Any]:
        return build_status()

    @api.get("/api/status")
    async def status() -> dict[str, Any]:
        return build_status()

    @api.post("/api/query")
    async def query(payload: dict[str, Any]) -> dict[str, Any]:
        result = await process_query(payload)
        if result.get("type") == "error":
            raise HTTPException(status_code=400, detail=result["message"])
        return result

    @api.websocket("/ws")
    async def websocket_endpoint(socket: WebSocket) -> None:
        await socket.accept()
        try:
            while True:
                payload = await socket.receive_json()
                message_type = payload.get("type")
                if message_type == "start_camera":
                    state["camera_connected"] = True
                    state["first_scene_narrated"] = False
                    await socket.send_json({"type": "status", "stage_status": build_status()})
                elif message_type == "stop_camera":
                    state["camera_connected"] = False
                    state["first_scene_narrated"] = False
                    assistant = state["assistant"]
                    if assistant is not None:
                        for component in (assistant.tracker, assistant.scene_memory, assistant.event_detector):
                            reset = getattr(component, "reset", None)
                            if callable(reset):
                                await run_in_threadpool(reset)
                    await socket.send_json({"type": "status", "stage_status": build_status()})
                elif message_type == "frame":
                    if not state["camera_connected"]:
                        await socket.send_json(
                            _error_payload(
                                "camera",
                                "camera_not_started",
                                "Camera session is not started",
                                recoverable=True,
                            )
                        )
                        continue
                    await socket.send_json(await process_frame_message(payload))
                elif message_type == "query":
                    await socket.send_json(await process_query(payload))
                elif message_type == "ping":
                    await socket.send_json({"type": "pong", "ts": payload.get("ts")})
                elif message_type == "pong":
                    continue
                elif message_type == "control":
                    action = payload.get("action")
                    if action == "pause":
                        state["camera_connected"] = False
                        await socket.send_json({"type": "status", "state": "paused", "message": "Camera paused"})
                    elif action == "resume":
                        state["camera_connected"] = True
                        await socket.send_json({"type": "status", "state": "streaming", "message": "Camera resumed"})
                    elif action == "stop":
                        state["camera_connected"] = False
                        await socket.send_json({"type": "status", "state": "idle", "message": "Camera stopped"})
                    else:
                        await socket.send_json(
                            _error_payload(
                                "protocol",
                                "unsupported_control",
                                f"Unsupported control action: {action}",
                                recoverable=True,
                            )
                        )
                else:
                    await socket.send_json(
                        _error_payload(
                            "protocol",
                            "unsupported_message",
                            f"Unsupported message type: {message_type}",
                            recoverable=True,
                        )
                    )
        except WebSocketDisconnect:
            state["camera_connected"] = False

    return api


app = create_api_app()