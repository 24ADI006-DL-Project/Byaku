from __future__ import annotations

import argparse
import asyncio
import base64
import json
import sys

import cv2
import websockets


def capture_frame(camera_index: int, max_width: int) -> tuple[str, tuple[int, int]]:
    camera = cv2.VideoCapture(camera_index)
    if not camera.isOpened():
        raise RuntimeError(f"Could not open webcam {camera_index}")
    try:
        ok, frame = camera.read()
        if not ok or frame is None:
            raise RuntimeError("Webcam opened but did not return a frame")
    finally:
        camera.release()

    height, width = frame.shape[:2]
    if width > max_width:
        scale = max_width / width
        frame = cv2.resize(frame, (max_width, round(height * scale)), interpolation=cv2.INTER_AREA)
    ok, encoded = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 78])
    if not ok:
        raise RuntimeError("Could not encode the webcam frame")
    return base64.b64encode(encoded.tobytes()).decode("ascii"), (frame.shape[1], frame.shape[0])


async def run(backend_url: str, camera_index: int, max_width: int) -> dict:
    encoded_frame, dimensions = capture_frame(camera_index, max_width)
    async with websockets.connect(backend_url, open_timeout=30, close_timeout=5) as socket:
        await socket.send(json.dumps({"type": "start_camera"}))
        start_status = json.loads(await socket.recv())
        if start_status.get("type") != "status":
            raise RuntimeError(f"Backend did not accept camera session: {start_status}")

        await socket.send(json.dumps({
            "type": "frame",
            "frame_id": 1,
            "width": dimensions[0],
            "height": dimensions[1],
            "data": encoded_frame,
        }))
        result = json.loads(await socket.recv())
        if result.get("type") != "frame_result":
            raise RuntimeError(f"Backend rejected webcam frame: {result}")

        stages = result.get("stage_status", {})
        report = {
            "frame_size": result.get("frame_size"),
            "processing_ms": result.get("processing_ms"),
            "status": result.get("status"),
            "detections": result.get("detections", []),
            "tracks": result.get("tracks", []),
            "objects": result.get("objects", []),
            "depth_available": result.get("depth_available"),
            "stage_status": stages,
        }
        required_stages = ("detection", "tracking", "depth", "spatial")
        report["pipeline_passed"] = all(stages.get(stage) == "running" for stage in required_stages)
        return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Send one real webcam frame through BYAKU's Python pipeline")
    parser.add_argument("--backend", default="ws://127.0.0.1:8765/ws")
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--max-width", type=int, default=640)
    args = parser.parse_args()
    try:
        report = asyncio.run(run(args.backend, args.camera, args.max_width))
    except Exception as exc:
        print(f"pipeline_test_failed: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(report, indent=2))
    return 0 if report["pipeline_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())