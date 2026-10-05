"""
Step 22 -- Real AI Inference End-to-End Tests.

Part 1: Direct API Test (FastAPI /ws with real YOLO and Depth Anything V2).
Part 2: Live Node Relay Test (Client -> Node.js (3001) -> FastAPI (8766)).

Uses real test1.jpg (no mock adapters, real weights, real local HF cache).
"""
from __future__ import annotations

import base64
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import threading
import time
import urllib.request
from typing import Any

WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

import cv2
import pytest
from fastapi.testclient import TestClient

from byaku_rebuild import api


def load_test_image_b64(image_name: str = "test1.jpg") -> tuple[str, int, int]:
    image_path = WORKSPACE_ROOT / image_name
    assert image_path.is_file(), f"Image {image_name} not found"
    img = cv2.imread(str(image_path))
    assert img is not None
    h, w = img.shape[:2]
    with open(image_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("ascii")
    return b64, w, h


# ===========================================================================
# Part 1: Direct API Test (Real Models)
# ===========================================================================

def test_direct_api_real_inference_frame_and_query() -> None:
    """Direct API test: WebSocket /ws with real YOLO and Depth Anything V2."""
    # Ensure no mock adapters are present on app.state
    if hasattr(api.app.state, "detection_adapter"):
        delattr(api.app.state, "detection_adapter")
    if hasattr(api.app.state, "depth_adapter"):
        delattr(api.app.state, "depth_adapter")
    if hasattr(api.app.state, "orchestrator"):
        delattr(api.app.state, "orchestrator")

    b64_data, w, h = load_test_image_b64("test1.jpg")

    with TestClient(api.app) as client:
        with client.websocket_connect("/ws") as ws:
            # 1. Initial status
            initial = ws.receive_json()
            assert initial.get("type") == "status"

            # 2. Start camera
            ws.send_json({"type": "start_camera"})
            status = ws.receive_json()
            assert status.get("type") == "status"
            assert status.get("camera") == "connected"

            # 3. Send real image frame
            t0 = time.perf_counter()
            ws.send_json({
                "type": "frame",
                "frame_id": 101,
                "width": w,
                "height": h,
                "data": b64_data,
            })
            frame_res = ws.receive_json()
            frame_time_ms = (time.perf_counter() - t0) * 1000.0

            assert frame_res.get("type") == "frame_result"
            assert frame_res.get("frame_id") == 101, "Browser frame_id must be echoed"
            assert frame_res.get("status") == "success"

            # Check real detections
            detections = frame_res.get("detections", [])
            assert len(detections) > 0, "Real YOLO should have found detections"
            detected_classes = {d["class_name"] for d in detections}
            assert "chair" in detected_classes or "bed" in detected_classes

            # Check real depth status
            stage_status = frame_res.get("stage_status", {})
            assert stage_status.get("detection") in {"ready", "running"}
            assert stage_status.get("depth") == "ready"

            # Check relationships and tracks
            assert len(frame_res.get("relationships", [])) > 0
            assert len(frame_res.get("tracks", [])) > 0

            # 4. Send query on real scene
            t0 = time.perf_counter()
            ws.send_json({
                "type": "query",
                "text": "What is to the left of the bed?",
            })
            query_res = ws.receive_json()
            query_time_ms = (time.perf_counter() - t0) * 1000.0

            assert query_res.get("type") == "query_result"
            assert "query" in query_res
            assert query_res.get("status") in {"visible", "ambiguous", "not_found"}
            assert "reasoning" in query_res
            assert len(query_res["reasoning"].get("objects", [])) > 0


# ===========================================================================
# Part 2: Live Node.js Relay E2E Test (Real Models)
# ===========================================================================

FASTAPI_HOST = "127.0.0.1"
FASTAPI_PORT = 8766
NODE_PORT = 3001


def _wait_tcp(port: int, timeout: float = 12.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            with socket.create_connection((FASTAPI_HOST, port), timeout=1.0):
                return True
        except OSError:
            time.sleep(0.2)
    return False


def run_node_relay_real_inference_e2e() -> dict[str, Any]:
    """Starts FastAPI (real inference) + Node relay, tests real frame and query through relay."""
    import uvicorn
    import websockets.sync.client as ws_sync

    # Ensure clean app state
    if hasattr(api.app.state, "detection_adapter"):
        delattr(api.app.state, "detection_adapter")
    if hasattr(api.app.state, "depth_adapter"):
        delattr(api.app.state, "depth_adapter")
    if hasattr(api.app.state, "orchestrator"):
        delattr(api.app.state, "orchestrator")

    # 1. Start FastAPI in thread
    server_thread = threading.Thread(
        target=lambda: uvicorn.run(
            api.app,
            host=FASTAPI_HOST,
            port=FASTAPI_PORT,
            log_level="warning",
        ),
        daemon=True,
    )
    server_thread.start()

    if not _wait_tcp(FASTAPI_PORT, timeout=15.0):
        raise RuntimeError(f"FastAPI failed to start on port {FASTAPI_PORT}")

    # 2. Start Node relay
    web_dir = WORKSPACE_ROOT / "web"
    env = os.environ.copy()
    env["PORT"] = str(NODE_PORT)
    env["PYTHON_BACKEND_URL"] = f"http://{FASTAPI_HOST}:{FASTAPI_PORT}"
    env["BYAKU_PYTHON_BACKEND_URL"] = f"http://{FASTAPI_HOST}:{FASTAPI_PORT}"

    node_cmd = ["node", "server/server.js"]
    node_proc = subprocess.Popen(
        node_cmd,
        cwd=str(web_dir),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    try:
        if not _wait_tcp(NODE_PORT, timeout=12.0):
            raise RuntimeError(f"Node relay failed to start on port {NODE_PORT}")

        b64_data, w, h = load_test_image_b64("test1.jpg")

        ws_url = f"ws://{FASTAPI_HOST}:{NODE_PORT}/ws"
        with ws_sync.connect(ws_url, max_size=10 * 1024 * 1024, open_timeout=10) as ws:
            # Receive initial status
            init_msg = json.loads(ws.recv(timeout=10))
            assert init_msg.get("type") == "status"

            # Start camera
            ws.send(json.dumps({"type": "start_camera"}))
            cam_msg = json.loads(ws.recv(timeout=10))
            assert cam_msg.get("type") == "status"
            assert cam_msg.get("camera") == "connected"

            # Submit real image frame
            t0 = time.perf_counter()
            ws.send(json.dumps({
                "type": "frame",
                "frame_id": 201,
                "width": w,
                "height": h,
                "data": b64_data,
            }))
            frame_raw = ws.recv(timeout=30)
            frame_time_ms = (time.perf_counter() - t0) * 1000.0
            frame_res = json.loads(frame_raw)

            assert frame_res.get("type") == "frame_result"
            assert frame_res.get("frame_id") == 201, "Frame ID must be echoed through relay"
            assert frame_res.get("status") == "success"

            detections = frame_res.get("detections", [])
            assert len(detections) > 0
            detected_classes = [d["class_name"] for d in detections]

            stage_status = frame_res.get("stage_status", {})
            assert stage_status.get("detection") in {"ready", "running"}
            assert stage_status.get("depth") == "ready"

            # Submit second frame in sequence (test2.jpg)
            b64_data_2, w2, h2 = load_test_image_b64("test2.jpg")
            t0 = time.perf_counter()
            ws.send(json.dumps({
                "type": "frame",
                "frame_id": 202,
                "width": w2,
                "height": h2,
                "data": b64_data_2,
            }))
            frame2_raw = ws.recv(timeout=30)
            frame2_time_ms = (time.perf_counter() - t0) * 1000.0
            frame2_res = json.loads(frame2_raw)

            assert frame2_res.get("type") == "frame_result"
            assert frame2_res.get("frame_id") == 202
            assert frame2_res.get("status") in {"success", "partial"}
            detections_2 = frame2_res.get("detections", [])
            assert len(detections_2) > 0
            assert len(frame2_res.get("tracks", [])) > 0

            # Query real scene
            t0 = time.perf_counter()
            ws.send(json.dumps({
                "type": "query",
                "text": "What is to the left of the bed?",
            }))
            query_raw = ws.recv(timeout=15)
            query_time_ms = (time.perf_counter() - t0) * 1000.0
            query_res = json.loads(query_raw)

            assert query_res.get("type") == "query_result"
            assert "answer" in query_res
            assert query_res.get("status") in {"visible", "ambiguous", "not_found"}

            return {
                "frame_id": frame_res["frame_id"],
                "frame_time_ms": frame_time_ms,
                "frame2_id": frame2_res["frame_id"],
                "frame2_time_ms": frame2_time_ms,
                "detections_count": len(detections),
                "classes": detected_classes,
                "detections_2_count": len(detections_2),
                "classes_2": [d["class_name"] for d in detections_2],
                "stage_status": stage_status,
                "query": query_res["query"],
                "answer": query_res["answer"],
                "query_status": query_res["status"],
                "query_time_ms": query_time_ms,
            }
    finally:
        node_proc.kill()
        node_proc.wait(timeout=5)


@pytest.mark.skipif(
    not shutil.which("node"),
    reason="node executable is not on PATH (sandbox mode)",
)
def test_node_relay_real_inference_e2e() -> None:
    """End-to-end test of real AI inference through the Node.js relay."""
    result = run_node_relay_real_inference_e2e()
    assert result["frame_id"] == 201
    assert result["frame2_id"] == 202
    assert result["detections_count"] > 0
    assert result["query_status"] in {"visible", "ambiguous", "not_found"}



if __name__ == "__main__":
    print("--- Running Direct API Test (Real Inference) ---")
    test_direct_api_real_inference_frame_and_query()
    print("PASS - Direct API with real models succeeded.")

    print("\n--- Running Live Node Relay E2E Test (Real Inference) ---")
    out = run_node_relay_real_inference_e2e()
    print(f"PASS - Node Relay E2E verified!")
    print(f"Frame 1 (ID: {out['frame_id']}): latency {out['frame_time_ms']:.1f}ms, detected {out['detections_count']} objects: {out['classes']}")
    print(f"Frame 2 (ID: {out['frame2_id']}): latency {out['frame2_time_ms']:.1f}ms, detected {out['detections_2_count']} objects: {out['classes_2']}")
    print(f"Stage Status: {out['stage_status']}")
    print(f"Query: '{out['query']}' (latency: {out['query_time_ms']:.1f}ms)")
    print(f"Answer: {out['answer']}")
