"""
Step 21 -- Live Node relay integration test script.

Starts FastAPI in a background thread (using uvicorn), then starts the real
Node.js relay process, then connects through Node to FastAPI via WebSocket,
sends a ping, verifies the pong returns through the relay path.

This script must be run with 'node' available on PATH. Run from the workspace root:
  python byaku_rebuild/tests/run_node_relay_test.py

Expected output (PASS):
  [FastAPI] Starting on 127.0.0.1:8765...
  [FastAPI] Healthy: {"service": "byaku-rebuild-api", "status": "running"}
  [Node]    Starting relay on port 3001, pointing at http://127.0.0.1:8765
  [WS]      Connecting through Node relay at ws://127.0.0.1:3001/ws ...
  [WS]      Received initial status: {"type": "status", ...}
  [WS]      Sending ping {"type": "ping", "ts": 42}
  [WS]      Received pong: {"type": "pong", "ts": 42}
  [RESULT]  PASS - ping/pong routed successfully through Node relay -> FastAPI
"""
from __future__ import annotations

import json
import os
import shutil
import signal
import subprocess
import sys
import threading
import time
import urllib.request

import uvicorn

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
FASTAPI_HOST = "127.0.0.1"
FASTAPI_PORT = 8766  # use 8766 to avoid collision with a running instance
NODE_PORT = 3001     # use 3001 to avoid collision with a running Node instance


def _start_fastapi() -> None:
    """Run uvicorn in this thread (blocking)."""
    sys.path.insert(0, ROOT)
    from byaku_rebuild import api
    uvicorn.run(api.app, host=FASTAPI_HOST, port=FASTAPI_PORT, log_level="warning")


def _wait_for_fastapi(timeout: float = 10.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(
                f"http://{FASTAPI_HOST}:{FASTAPI_PORT}/health", timeout=1
            ) as resp:
                body = json.loads(resp.read())
                if body.get("status") == "running":
                    return True
        except Exception:
            time.sleep(0.25)
    return False


def _run_node_relay(node_bin: str, node_port: int, backend_url: str) -> "subprocess.Popen[bytes]":
    env = os.environ.copy()
    env["PORT"] = str(node_port)
    env["PYTHON_BACKEND_URL"] = backend_url
    server_js = os.path.join(ROOT, "web", "server", "server.js")
    return subprocess.Popen(
        [node_bin, server_js],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )


def _wait_for_node(port: int, timeout: float = 10.0) -> bool:
    """Wait until Node's TCP port is accepting connections."""
    import socket

    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=1):
                return True
        except OSError:
            time.sleep(0.25)
    return False


def main() -> int:
    node_bin = shutil.which("node")
    if node_bin is None:
        print("[BLOCKED] 'node' not found on PATH. Cannot run live relay test.")
        return 1

    # Start FastAPI in a daemon thread
    print(f"[FastAPI] Starting on {FASTAPI_HOST}:{FASTAPI_PORT}...")
    t = threading.Thread(target=_start_fastapi, daemon=True)
    t.start()

    if not _wait_for_fastapi():
        print("[FastAPI] ERROR: did not become healthy within 10 seconds.")
        return 1
    print(f"[FastAPI] Healthy. Endpoint: http://{FASTAPI_HOST}:{FASTAPI_PORT}/health")

    backend_url = f"http://{FASTAPI_HOST}:{FASTAPI_PORT}"
    print(f"[Node]    Starting relay on port {NODE_PORT}, backend={backend_url}")
    node_proc = _run_node_relay(node_bin, NODE_PORT, backend_url)

    if not _wait_for_node(NODE_PORT):
        print("[Node]    ERROR: relay TCP port did not open within 10 seconds.")
        node_proc.kill()
        try:
            stdout, stderr = node_proc.communicate(timeout=3)
            print(f"[Node]    stdout: {stdout.decode(errors='replace')[:500]}")
            print(f"[Node]    stderr: {stderr.decode(errors='replace')[:500]}")
        except Exception:
            pass
        return 1

    print(f"[Node]    Ready. Relay at http://127.0.0.1:{NODE_PORT}")

    # Connect through Node WebSocket relay
    try:
        import websocket  # websocket-client
    except ImportError:
        # Fall back to using the websockets library (already installed as dep of fastapi)
        import asyncio
        import websockets.sync.client as wsclient

        print(f"[WS]      Connecting through Node relay at ws://127.0.0.1:{NODE_PORT}/ws ...")
        result = 1
        try:
            with wsclient.connect(f"ws://127.0.0.1:{NODE_PORT}/ws") as ws:
                initial = json.loads(ws.recv())
                print(f"[WS]      Received initial message: type={initial.get('type')!r}")
                assert initial.get("type") == "status", f"Expected status, got {initial}"

                ws.send(json.dumps({"type": "ping", "ts": 42}))
                pong_raw = ws.recv()
                pong = json.loads(pong_raw)
                print(f"[WS]      Received: {pong}")
                assert pong.get("type") == "pong", f"Expected pong, got {pong}"
                assert pong.get("ts") == 42, f"ts mismatch: {pong}"

                print("[RESULT]  PASS - ping/pong routed through Node relay -> FastAPI")
                result = 0
        except Exception as exc:
            print(f"[RESULT]  FAIL - {exc}")
            result = 1
        finally:
            node_proc.terminate()
        return result

    # websocket-client path (less likely but kept as fallback)
    print(f"[WS]      Connecting through Node relay at ws://127.0.0.1:{NODE_PORT}/ws ...")
    ws = websocket.WebSocket()
    result = 1
    try:
        ws.connect(f"ws://127.0.0.1:{NODE_PORT}/ws")
        initial = json.loads(ws.recv())
        print(f"[WS]      Received initial message: type={initial.get('type')!r}")
        assert initial.get("type") == "status"
        ws.send(json.dumps({"type": "ping", "ts": 42}))
        pong = json.loads(ws.recv())
        print(f"[WS]      Received: {pong}")
        assert pong.get("type") == "pong" and pong.get("ts") == 42
        print("[RESULT]  PASS - ping/pong routed through Node relay -> FastAPI")
        result = 0
    except Exception as exc:
        print(f"[RESULT]  FAIL - {exc}")
    finally:
        ws.close()
        node_proc.terminate()
    return result


if __name__ == "__main__":
    sys.exit(main())
