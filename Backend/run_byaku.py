"""
Starts FastAPI (8765) and Node.js relay (3000) with real AI inference,
pre-populates real frame data from bedroom_scene_desk.jpg and real query reasoning,
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
from typing import Any

WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

import cv2
import websockets.sync.client as ws_sync
from Backend import fastapi_application as api


FASTAPI_HOST = "127.0.0.1"
FASTAPI_PORT = 8765
NODE_PORT = 3000
INITIAL_FRAME_TIMEOUT_ENV = "BYAKU_INITIAL_FRAME_TIMEOUT_SECONDS"
DEFAULT_INITIAL_FRAME_TIMEOUT_SECONDS = 180
MAX_INITIAL_FRAME_TIMEOUT_SECONDS = 300


def _initial_frame_timeout_seconds() -> int:
    configured_timeout = os.environ.get(
        INITIAL_FRAME_TIMEOUT_ENV,
        str(DEFAULT_INITIAL_FRAME_TIMEOUT_SECONDS),
    )
    try:
        timeout_seconds = int(configured_timeout)
    except ValueError as error:
        raise ValueError(f"{INITIAL_FRAME_TIMEOUT_ENV} must be an integer number of seconds") from error
    return min(max(timeout_seconds, 1), MAX_INITIAL_FRAME_TIMEOUT_SECONDS)


def _receive_initial_frame(websocket: Any) -> Any:
    return websocket.recv(timeout=_initial_frame_timeout_seconds())


def _wait_tcp(port: int, timeout: float = 12.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            with socket.create_connection((FASTAPI_HOST, port), timeout=0.5):
                return True
        except OSError:
            time.sleep(0.2)
    return False


def _resolve_frontend_dir() -> Path:
    for candidate in (WORKSPACE_ROOT / "UI", WORKSPACE_ROOT / "web-redesign"):
        if candidate.is_dir():
            return candidate
    raise FileNotFoundError(
        "No frontend directory found. Expected either 'UI/' (current static frontend) or 'web-redesign/' (legacy Vite app)."
    )


def _start_frontend(frontend_dir: Path, env: dict[str, str]) -> subprocess.Popen[Any]:
    if frontend_dir.name == "web-redesign":
        vite_cli = frontend_dir / "node_modules" / "vite" / "bin" / "vite.js"
        if not vite_cli.is_file():
            raise FileNotFoundError(f"Vite is not installed in {frontend_dir}; install the web-redesign dependencies first.")
        build = subprocess.run(["node", str(vite_cli), "build"], cwd=str(frontend_dir), env=env, check=False)
        if build.returncode != 0:
            raise RuntimeError("redesigned frontend build failed")
        proc = subprocess.Popen(
            ["node", str(vite_cli), "preview", "--host", FASTAPI_HOST, "--port", str(NODE_PORT), "--strictPort"],
            cwd=str(frontend_dir),
            env=env,
        )
    else:
        relay_script = frontend_dir / "server" / "http_websocket_relay.js"
        if not relay_script.is_file():
            raise FileNotFoundError(f"Frontend relay script not found: {relay_script}")
        proc = subprocess.Popen(
            ["node", str(relay_script)],
            cwd=str(frontend_dir),
            env={**env, "PORT": str(NODE_PORT)},
        )
    if not _wait_tcp(NODE_PORT, timeout=12.0):
        proc.kill()
        raise RuntimeError(f"frontend failed to start on port {NODE_PORT}")
    return proc


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

    # 2. Start the browser frontend using the current repo layout (UI/ relay) or legacy Vite app
    env = os.environ.copy()
    env["PYTHON_BACKEND_URL"] = f"http://{FASTAPI_HOST}:{FASTAPI_PORT}"
    env["BYAKU_PYTHON_BACKEND_URL"] = f"http://{FASTAPI_HOST}:{FASTAPI_PORT}"

    try:
        frontend_dir = _resolve_frontend_dir()
        node_proc = _start_frontend(frontend_dir, env)
    except Exception as exc:
        print(f"ERROR: {exc}")
        sys.exit(1)

    print(f"[Frontend] Running on http://{FASTAPI_HOST}:{NODE_PORT} via {frontend_dir.name}")

    # 3. Populate live scene with real inference
    try:
        image_path = WORKSPACE_ROOT / "Data" / "samples" / "bedroom_scene_desk.jpg"
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
            _receive_initial_frame(ws)  # frame result; cold-start allowance is bounded and configurable.
            ws.send(json.dumps({
                "type": "query",
                "text": "What is to the left of the bed?",
            }))
            ws.recv(timeout=15)  # query result
        print("[Populate] Scene populated with real bedroom_scene_desk.jpg detections and query reasoning.")
    except Exception as e:
        print(f"[Populate] Warning during pre-population: {e}")

    print("[Ready] Both servers active and ready for browser validation. Press Ctrl+C or kill process to exit.")
    try:
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        pass
    finally:
        print("[Shutdown] Stopping frontend process...")
        node_proc.kill()
        node_proc.wait(timeout=5)
        print("[Shutdown] Done.")


if __name__ == "__main__":
    main()
