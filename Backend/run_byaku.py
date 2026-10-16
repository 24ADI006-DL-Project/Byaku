"""
Starts FastAPI (8765) and Node.js relay (3000) with real AI inference,
pre-populates real frame data from bedroom_scene_desk.jpg and real query reasoning,
and keeps the servers running for browser inspection until interrupted.
"""
from __future__ import annotations

import base64
from collections.abc import Mapping
import json
import math
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import tempfile
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


def _evaluate_model_accuracy(
    model: Any,
    weights_path: Path,
    *,
    dataset_root: Path | None = None,
) -> dict[str, Any]:
    from Workflow.training.run_detector_training import (
        BATCH_SIZE,
        DEFAULT_OUTPUT_ROOT,
        IMAGE_SIZE,
        MAX_DETECTIONS,
        VALIDATION_CONFIDENCE,
        VALIDATION_IOU,
        WORKERS,
        _write_baseline_test_dataset,
    )

    dataset_root = Path(dataset_root or DEFAULT_OUTPUT_ROOT).resolve()
    if not (dataset_root / "data.yaml").is_file():
        raise FileNotFoundError(f"prepared held-out dataset not found: {dataset_root / 'data.yaml'}")
    model_names = getattr(model, "names", None)
    if not isinstance(model_names, Mapping):
        raise ValueError("loaded YOLO model does not expose class names for test label mapping")

    test_images = sorted(
        path for path in (dataset_root / "images" / "test").glob("*")
        if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )
    if not test_images:
        raise ValueError(f"held-out test split is empty: {dataset_root / 'images' / 'test'}")

    with tempfile.TemporaryDirectory(prefix="byaku-model-accuracy-") as temp_dir:
        baseline_data, class_support = _write_baseline_test_dataset(
            dataset_root=dataset_root,
            baseline_dataset_root=Path(temp_dir) / "baseline_test_dataset",
            baseline_names={int(key): str(value) for key, value in model_names.items()},
        )
        result = model.val(
            data=str(baseline_data),
            split="test",
            imgsz=IMAGE_SIZE,
            batch=BATCH_SIZE,
            device="cpu",
            workers=WORKERS,
            conf=VALIDATION_CONFIDENCE,
            iou=VALIDATION_IOU,
            max_det=MAX_DETECTIONS,
            plots=False,
            save_json=False,
            verbose=False,
            project=temp_dir,
            name="validation",
            exist_ok=False,
        )

    result_metrics = result.results_dict
    metric_names = {
        "precision": "metrics/precision(B)",
        "recall": "metrics/recall(B)",
        "mAP50": "metrics/mAP50(B)",
        "mAP50_95": "metrics/mAP50-95(B)",
    }
    metrics = {name: float(result_metrics[key]) for name, key in metric_names.items()}
    if not all(math.isfinite(value) for value in metrics.values()):
        raise ValueError("YOLO validation returned non-finite accuracy metrics")
    return {
        "checkpoint": str(Path(weights_path).resolve()),
        "dataset": str(dataset_root),
        "test_images": len(test_images),
        "test_boxes": sum(class_support.values()),
        "evaluated_classes": len(class_support),
        **metrics,
    }


def _print_model_accuracy(
    model: Any,
    weights_path: Path,
    *,
    dataset_root: Path | None = None,
) -> None:
    print(
        f"[Model Accuracy] Evaluating {weights_path} on the held-out test split...",
        flush=True,
    )
    try:
        metrics = _evaluate_model_accuracy(model, weights_path, dataset_root=dataset_root)
    except (FileNotFoundError, OSError, RuntimeError, ValueError, KeyError, TypeError, AttributeError) as error:
        print(
            f"[Model Accuracy] UNAVAILABLE: {type(error).__name__}: {error}",
            file=sys.stderr,
            flush=True,
        )
        return
    print(
        "[Model Accuracy] "
        f"checkpoint={metrics['checkpoint']} | "
        f"test={metrics['test_images']} images, {metrics['test_boxes']} boxes, "
        f"{metrics['evaluated_classes']} supported classes | "
        f"Precision={metrics['precision']:.2%}, "
        f"Recall={metrics['recall']:.2%}, "
        f"mAP@0.5={metrics['mAP50']:.2%}, "
        f"mAP@0.5:0.95={metrics['mAP50_95']:.2%}",
        flush=True,
    )


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

    from Workflow.vision.object_detector_tracker import DEFAULT_WEIGHTS, DetectionTrackingAdapter

    detection_adapter = DetectionTrackingAdapter(weights_path=DEFAULT_WEIGHTS)
    api.app.state.detection_adapter = detection_adapter
    _print_model_accuracy(detection_adapter.model, DEFAULT_WEIGHTS)

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
