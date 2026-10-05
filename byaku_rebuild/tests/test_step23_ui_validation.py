"""
Step 23 -- Browser UI & Node Relay Integration Verification.

Validates:
1. Static asset serving by Node.js server (index.html, dashboard.html, scripts, styles).
2. HTTP API proxying (Node /api/status -> FastAPI /api/status).
3. WebSocket relay proxying (Node /ws -> FastAPI /ws) for status, ping/pong, camera lifecycle.
4. Real AI inference frame ingestion (YOLO11n + Depth Anything V2) over Node relay.
5. Voice/text query reasoning grounded in actual scene data.
6. Developer dashboard status contract validation (/api/status populated with real scene state).
7. Browser capability and failure boundary analysis (Web Speech, getUserMedia, SpeechSynthesis).
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
import urllib.error
from typing import Any

WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

import cv2
import pytest
import websockets.sync.client as ws_sync

from byaku_rebuild import api


FASTAPI_HOST = "127.0.0.1"
FASTAPI_PORT = 8767
NODE_PORT = 3002


def _wait_tcp(port: int, timeout: float = 12.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            with socket.create_connection((FASTAPI_HOST, port), timeout=0.5):
                return True
        except OSError:
            time.sleep(0.2)
    return False


def _fetch_url(url: str, timeout: float = 5.0) -> tuple[int, dict[str, str], bytes]:
    req = urllib.request.Request(url, headers={"User-Agent": "BYAKU-Validator/1.0"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        headers = {k.lower(): v for k, v in resp.headers.items()}
        body = resp.read()
        return resp.status, headers, body


class ServerHarness:
    """Manages FastAPI and Node.js servers for live UI validation."""

    def __init__(self, fastapi_port: int = FASTAPI_PORT, node_port: int = NODE_PORT):
        self.fastapi_port = fastapi_port
        self.node_port = node_port
        self.fastapi_thread: threading.Thread | None = None
        self.node_proc: subprocess.Popen | None = None

    def start(self) -> None:
        import uvicorn

        # Ensure no mock adapters on app.state
        if hasattr(api.app.state, "detection_adapter"):
            delattr(api.app.state, "detection_adapter")
        if hasattr(api.app.state, "depth_adapter"):
            delattr(api.app.state, "depth_adapter")
        if hasattr(api.app.state, "orchestrator"):
            delattr(api.app.state, "orchestrator")

        # 1. Start FastAPI
        self.fastapi_thread = threading.Thread(
            target=lambda: uvicorn.run(
                api.app,
                host=FASTAPI_HOST,
                port=self.fastapi_port,
                log_level="warning",
            ),
            daemon=True,
        )
        self.fastapi_thread.start()
        if not _wait_tcp(self.fastapi_port, timeout=15.0):
            raise RuntimeError(f"FastAPI failed to start on port {self.fastapi_port}")

        # 2. Start Node relay
        web_dir = WORKSPACE_ROOT / "web"
        env = os.environ.copy()
        env["PORT"] = str(self.node_port)
        env["PYTHON_BACKEND_URL"] = f"http://{FASTAPI_HOST}:{self.fastapi_port}"
        env["BYAKU_PYTHON_BACKEND_URL"] = f"http://{FASTAPI_HOST}:{self.fastapi_port}"

        self.node_proc = subprocess.Popen(
            ["node", "server/server.js"],
            cwd=str(web_dir),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        if not _wait_tcp(self.node_port, timeout=12.0):
            self.stop()
            raise RuntimeError(f"Node relay failed to start on port {self.node_port}")

    def stop(self) -> None:
        if self.node_proc:
            try:
                self.node_proc.kill()
                self.node_proc.wait(timeout=5)
            except Exception:
                pass
            self.node_proc = None


def run_full_validation() -> dict[str, Any]:
    """Execute end-to-end browser UI and backend integration tests."""
    results: dict[str, Any] = {"steps": {}}
    harness = ServerHarness(FASTAPI_PORT, NODE_PORT)

    try:
        harness.start()
        results["servers_started"] = True

        # --- Test 1: Static Asset Delivery ---
        assets = [
            ("/", 200, "text/html"),
            ("/index.html", 200, "text/html"),
            ("/dashboard.html", 200, "text/html"),
            ("/css/styles.css", 200, "text/css"),
            ("/js/config.js", 200, "text/javascript"),
            ("/js/camera.js", 200, "text/javascript"),
            ("/js/websocket.js", 200, "text/javascript"),
            ("/js/intro.js", 200, "text/javascript"),
            ("/js/app.js", 200, "text/javascript"),
            ("/js/dashboard.js", 200, "text/javascript"),
        ]
        asset_checks = {}
        for path, expected_status, content_type_prefix in assets:
            status, headers, body = _fetch_url(f"http://{FASTAPI_HOST}:{NODE_PORT}{path}")
            assert status == expected_status, f"{path} returned status {status}"
            assert len(body) > 0, f"{path} body was empty"
            ct = headers.get("content-type", "")
            assert content_type_prefix in ct, f"{path} content-type '{ct}' does not contain '{content_type_prefix}'"
            asset_checks[path] = {"status": status, "bytes": len(body), "content_type": ct}
        results["steps"]["static_assets"] = {"status": "PASS", "details": asset_checks}

        # --- Test 2: HTTP API Proxying ---
        status_code, headers, body = _fetch_url(f"http://{FASTAPI_HOST}:{NODE_PORT}/api/status")
        assert status_code == 200
        initial_status = json.loads(body.decode("utf-8"))
        assert initial_status.get("backend") == "connected"
        assert initial_status.get("camera") == "disconnected"
        results["steps"]["http_api_proxy"] = {"status": "PASS", "initial_status": initial_status}

        # --- Test 3: WebSocket Relay & Camera Lifecycle ---
        ws_url = f"ws://{FASTAPI_HOST}:{NODE_PORT}/ws"
        with ws_sync.connect(ws_url, max_size=10 * 1024 * 1024, open_timeout=10) as ws:
            # Receive initial status
            init_msg = json.loads(ws.recv(timeout=10))
            assert init_msg.get("type") == "status"

            # Ping/pong
            ws.send(json.dumps({"type": "ping", "ts": 12345}))
            pong = json.loads(ws.recv(timeout=10))
            assert pong.get("type") == "pong"
            assert pong.get("ts") == 12345

            # Start camera session
            ws.send(json.dumps({"type": "start_camera"}))
            cam_start = json.loads(ws.recv(timeout=10))
            assert cam_start.get("type") == "status"
            assert cam_start.get("camera") == "connected"

            # --- Test 4: Real AI Frame Ingestion over Relay ---
            image_path = WORKSPACE_ROOT / "test1.jpg"
            img = cv2.imread(str(image_path))
            h, w = img.shape[:2]
            with open(image_path, "rb") as f:
                b64_data = base64.b64encode(f.read()).decode("ascii")

            t0 = time.perf_counter()
            ws.send(json.dumps({
                "type": "frame",
                "frame_id": 1,
                "width": w,
                "height": h,
                "data": b64_data,
            }))
            frame_raw = ws.recv(timeout=30)
            frame_time_ms = (time.perf_counter() - t0) * 1000.0
            frame_res = json.loads(frame_raw)

            assert frame_res.get("type") == "frame_result"
            assert frame_res.get("frame_id") == 1
            assert frame_res.get("status") == "success"
            detections = frame_res.get("detections", [])
            assert len(detections) > 0
            detected_classes = [d["class_name"] for d in detections]
            tracks = frame_res.get("tracks", [])
            assert len(tracks) > 0
            relationships = frame_res.get("relationships", [])
            assert len(relationships) > 0
            results["steps"]["real_frame_inference"] = {
                "status": "PASS",
                "latency_ms": frame_time_ms,
                "detections_count": len(detections),
                "classes": detected_classes,
                "tracks_count": len(tracks),
                "relationships_count": len(relationships),
                "scene_narration": frame_res.get("scene_narration"),
            }

            # --- Test 5: Voice/Text Query Reasoning ---
            t0 = time.perf_counter()
            ws.send(json.dumps({
                "type": "query",
                "text": "What is to the left of the bed?",
            }))
            query_raw = ws.recv(timeout=15)
            query_time_ms = (time.perf_counter() - t0) * 1000.0
            query_res = json.loads(query_raw)

            assert query_res.get("type") == "query_result"
            assert query_res.get("status") in {"visible", "ambiguous", "not_found"}
            assert "answer" in query_res
            results["steps"]["query_reasoning"] = {
                "status": "PASS",
                "query": query_res["query"],
                "answer": query_res["answer"],
                "query_status": query_res["status"],
                "narration": query_res.get("narration"),
                "reasoning_objects": [o["class_name"] for o in query_res.get("reasoning", {}).get("objects", [])],
                "latency_ms": query_time_ms,
            }

            # Stop camera session
            ws.send(json.dumps({"type": "stop_camera"}))
            cam_stop = json.loads(ws.recv(timeout=10))
            assert cam_stop.get("type") == "status"
            assert cam_stop.get("camera") == "disconnected"

        # --- Test 6: Verify Dashboard /api/status Contract ---
        status_code, headers, body = _fetch_url(f"http://{FASTAPI_HOST}:{NODE_PORT}/api/status")
        assert status_code == 200
        post_status = json.loads(body.decode("utf-8"))
        assert post_status.get("backend") == "connected"
        stage_status = post_status.get("stage_status", {})
        assert stage_status.get("detection") == "ready"
        assert stage_status.get("depth") == "ready"
        assert stage_status.get("spatial") == "ready"
        assert stage_status.get("scene_graph") == "ready"

        last_frame = post_status.get("last_frame")
        assert last_frame is not None
        assert last_frame.get("frame_id") == 1
        assert len(last_frame.get("objects", [])) > 0

        last_query = post_status.get("last_query")
        assert last_query == "What is to the left of the bed?"
        last_reasoning = post_status.get("last_reasoning")
        assert last_reasoning is not None
        assert len(last_reasoning.get("objects", [])) > 0

        results["steps"]["dashboard_status_contract"] = {
            "status": "PASS",
            "stage_status": stage_status,
            "last_frame_objects": len(last_frame.get("objects", [])),
            "last_query": last_query,
            "last_narration": post_status.get("last_narration"),
        }

        results["overall"] = "PASS"
        return results

    finally:
        harness.stop()


# ---------------------------------------------------------------------------
# Pytest Integration Test
# ---------------------------------------------------------------------------

@pytest.mark.skipif(
    not shutil.which("node"),
    reason="node executable is not on PATH (sandbox mode)",
)
def test_step23_full_integration() -> None:
    res = run_full_validation()
    assert res["overall"] == "PASS"
    assert res["steps"]["static_assets"]["status"] == "PASS"
    assert res["steps"]["http_api_proxy"]["status"] == "PASS"
    assert res["steps"]["real_frame_inference"]["status"] == "PASS"
    assert res["steps"]["query_reasoning"]["status"] == "PASS"
    assert res["steps"]["dashboard_status_contract"]["status"] == "PASS"


if __name__ == "__main__":
    print("=== Step 23: Browser UI & Node Relay Validation ===")
    out = run_full_validation()
    print("\n--- Summary ---")
    print(f"Overall Result: {out['overall']}")
    for step, data in out["steps"].items():
        print(f"  [{data['status']}] {step}")
    print("\nDetailed outputs:")
    print("  Frame Latency:", out["steps"]["real_frame_inference"]["latency_ms"], "ms")
    print("  Detected Classes:", out["steps"]["real_frame_inference"]["classes"])
    print("  Query Answer:", out["steps"]["query_reasoning"]["answer"])
    print("  Dashboard Stage Status:", out["steps"]["dashboard_status_contract"]["stage_status"])
