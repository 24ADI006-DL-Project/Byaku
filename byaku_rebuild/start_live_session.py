"""
Starts FastAPI (8765) and Node.js relay (3000) with real AI inference,
pre-populates real frame data from test1.jpg and real query reasoning,
and keeps the servers running for browser inspection until interrupted.
"""
from __future__ import annotations

import base64
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import threading
import time

WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

import cv2
import websockets.sync.client as ws_sync
from byaku_rebuild import api


FASTAPI_HOST = "127.0.0.1"
FASTAPI_PORT = 8765
NODE_PORT = 3000


def _wait_tcp(port: int, timeout: float = 12.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            with socket.create_connection((FASTAPI_HOST, port), timeout=0.5):
                return True
        except OSError:
            time.sleep(0.2)
    return False


def main() -> None:
    import uvicorn

    # Clear any leftover mock adapters
    if hasattr(api.app.state, "detection_adapter"):
        delattr(api.app.state, "detection_adapter")
    if hasattr(api.app.state, "depth_adapter"):
        delattr(api.app.state, "depth_adapter")
    if hasattr(api.app.state, "orchestrator"):
        delattr(api.app.state, "orchestrator")

    # 1. Start FastAPI in daemon thread
    server_thread = threading.Thread(
        target=lambda: uvicorn.run(api.app, host=FASTAPI_HOST, port=FASTAPI_PORT, log_level="warning"),
        daemon=True,
    )
    server_thread.start()
    if not _wait_tcp(FASTAPI_PORT, timeout=15.0):
        print(f"ERROR: FastAPI failed to start on port {FASTAPI_PORT}")
        sys.exit(1)
    print(f"[FastAPI] Running on http://{FASTAPI_HOST}:{FASTAPI_PORT}")

    # 2. Start Node relay
    web_dir = WORKSPACE_ROOT / "web"
    env = os.environ.copy()
    env["PORT"] = str(NODE_PORT)
    env["PYTHON_BACKEND_URL"] = f"http://{FASTAPI_HOST}:{FASTAPI_PORT}"
    env["BYAKU_PYTHON_BACKEND_URL"] = f"http://{FASTAPI_HOST}:{FASTAPI_PORT}"

    node_proc = subprocess.Popen(
        ["node", "server/server.js"],
        cwd=str(web_dir),
        env=env,
    )
    if not _wait_tcp(NODE_PORT, timeout=12.0):
        print(f"ERROR: Node relay failed to start on port {NODE_PORT}")
        node_proc.kill()
        sys.exit(1)
    print(f"[Node]    Running on http://{FASTAPI_HOST}:{NODE_PORT}")

    # 3. Populate live scene with real inference
    try:
        image_path = WORKSPACE_ROOT / "test1.jpg"
        img = cv2.imread(str(image_path))
        h, w = img.shape[:2]
        with open(image_path, "rb") as f:
            b64_data = base64.b64encode(f.read()).decode("ascii")

        ws_url = f"ws://{FASTAPI_HOST}:{NODE_PORT}/ws"
        with ws_sync.connect(ws_url, max_size=10 * 1024 * 1024, open_timeout=10) as ws:
            ws.recv(timeout=10)  # initial status
            ws.send(json.dumps({"type": "start_camera"}))
            ws.recv(timeout=10)  # camera status
            ws.send(json.dumps({
                "type": "frame",
                "frame_id": 1,
                "width": w,
                "height": h,
                "data": b64_data,
            }))
            ws.recv(timeout=30)  # frame result
            ws.send(json.dumps({
                "type": "query",
                "text": "What is to the left of the bed?",
            }))
            ws.recv(timeout=15)  # query result
        print("[Populate] Scene populated with real test1.jpg detections and query reasoning.")
    except Exception as e:
        print(f"[Populate] Warning during pre-population: {e}")

    print("[Ready] Both servers active and ready for browser validation. Press Ctrl+C or kill process to exit.")
    try:
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        pass
    finally:
        print("[Shutdown] Stopping Node process...")
        node_proc.kill()
        node_proc.wait(timeout=5)
        print("[Shutdown] Done.")


if __name__ == "__main__":
    main()
