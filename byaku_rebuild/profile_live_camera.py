"""Measure real BYAKU browser-frame stages using a local video as camera input."""
from __future__ import annotations

import argparse
import base64
import json
from pathlib import Path
import time
from typing import Any

import cv2
from fastapi.testclient import TestClient

from byaku_rebuild.api import _ensure_live_adapters, app
from byaku_rebuild.frame_profiler import FrameProfiler


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_VIDEO = ROOT / "pinterest-video.mp4"


def _load_jpeg_frames(video_path: Path, count: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open benchmark video: {video_path}")
    frames: list[dict[str, Any]] = []
    try:
        metadata = {
            "video_fps": float(capture.get(cv2.CAP_PROP_FPS)),
            "video_frame_count": int(capture.get(cv2.CAP_PROP_FRAME_COUNT)),
            "video_width": int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
            "video_height": int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        }
        for frame_number in range(count):
            success, frame = capture.read()
            if not success or frame is None:
                raise RuntimeError(
                    f"Benchmark video ended after {len(frames)} frames; {count} were requested"
                )
            height, width = frame.shape[:2]
            encoded_ok, jpeg = cv2.imencode(".jpg", frame)
            if not encoded_ok:
                raise RuntimeError(f"Could not JPEG-encode local video frame {frame_number}")
            frames.append(
                {
                    "width": width,
                    "height": height,
                    "data": base64.b64encode(jpeg.tobytes()).decode("ascii"),
                }
            )
    finally:
        capture.release()
    return frames, metadata


def _measurements(values: list[float]) -> dict[str, float | int]:
    if not values:
        raise RuntimeError("No real measurements were collected")
    ordered = sorted(values)
    middle = len(ordered) // 2
    median = (
        ordered[middle]
        if len(ordered) % 2
        else (ordered[middle - 1] + ordered[middle]) / 2
    )
    return {
        "count": len(values),
        "mean_ms": sum(values) / len(values),
        "median_ms": median,
        "min_ms": ordered[0],
        "max_ms": ordered[-1],
    }


def run_benchmark(
    video_path: Path = DEFAULT_VIDEO,
    *,
    measured_frames: int = 10,
    warmup_frames: int = 2,
) -> dict[str, Any]:
    if measured_frames < 10:
        raise ValueError("measured_frames must be at least 10")
    if warmup_frames < 1:
        raise ValueError("warmup_frames must be positive")
    video_path = video_path.resolve()
    if not video_path.is_file():
        raise FileNotFoundError(f"Local benchmark video not found: {video_path}")

    frames, video_metadata = _load_jpeg_frames(video_path, measured_frames + warmup_frames)
    profiler = FrameProfiler()
    app.state.frame_profiler = profiler
    initialization_started = time.perf_counter()
    detection_adapter, depth_adapter, orchestrator = _ensure_live_adapters()
    initialization_ms = (time.perf_counter() - initialization_started) * 1000
    adapter_ids = (id(detection_adapter), id(depth_adapter), id(orchestrator))
    profiler.clear()

    round_trip_times: list[float] = []
    with TestClient(app) as client:
        with client.websocket_connect("/ws") as websocket:
            initial_status = websocket.receive_json()
            if initial_status.get("type") != "status":
                raise RuntimeError(f"Unexpected initial WebSocket message: {initial_status!r}")
            websocket.send_json({"type": "start_camera"})
            camera_status = websocket.receive_json()
            if camera_status.get("type") != "status":
                raise RuntimeError(f"Could not start browser-frame session: {camera_status!r}")

            for index, frame in enumerate(frames):
                if index == warmup_frames:
                    profiler.clear()
                client_frame_id = index + 1
                started = time.perf_counter()
                websocket.send_json(
                    {
                        "type": "frame",
                        "frame_id": client_frame_id,
                        "width": frame["width"],
                        "height": frame["height"],
                        "data": frame["data"],
                    }
                )
                response = websocket.receive_json()
                elapsed_ms = (time.perf_counter() - started) * 1000
                if response.get("type") != "frame_result":
                    raise RuntimeError(f"Frame {client_frame_id} failed: {response!r}")
                if response.get("status") not in {"success", "partial"}:
                    raise RuntimeError(
                        f"Frame {client_frame_id} returned unexpected status {response.get('status')!r}"
                    )
                if index >= warmup_frames:
                    round_trip_times.append(elapsed_ms)
                    profiler.record("frame_to_response_round_trip", elapsed_ms)

    profile = profiler.summary()
    stages = profile["stages"]
    expected_counts = {
        "websocket_receive": measured_frames,
        "websocket_json_parse": measured_frames,
        "jpeg_base64_decode_and_opencv_decode": measured_frames,
        "depth_anything_total": measured_frames,
        "yolo_bytetrack_total": measured_frames,
        "pipeline_orchestration_total": measured_frames,
        "backend_processing_total": measured_frames,
        "response_json_serialization": measured_frames,
        "websocket_send": measured_frames,
        "frame_to_response_round_trip": measured_frames,
    }
    for stage, expected_count in expected_counts.items():
        actual = stages.get(stage, {}).get("count", 0)
        if actual != expected_count:
            raise RuntimeError(f"Stage {stage!r} recorded {actual} of {expected_count} frames")

    means = {name: data["mean_ms"] for name, data in stages.items()}
    total_backend_ms = means["backend_processing_total"]
    depth_ms = means["depth_anything_total"]
    detection_ms = means["yolo_bytetrack_total"]
    elsewhere_pct = max(0.0, 100.0 - (depth_ms + detection_ms) / total_backend_ms * 100)
    candidates = [
        ("Depth Anything V2", depth_ms),
        ("JPEG/base64 decode", means["jpeg_base64_decode_and_opencv_decode"]),
        ("Pipeline orchestration", means.get("pipeline_orchestration_total", 0.0)),
        ("Depth preprocessing/conversion", means.get("depth_preprocessing", 0.0)),
        ("YOLO + ByteTrack", detection_ms),
        ("Spatial relationships", means.get("spatial_relationships", 0.0)),
        ("Scene memory", means.get("scene_memory", 0.0)),
        ("Scene graph", means.get("scene_graph", 0.0)),
        ("Scene description/narration", means.get("scene_description", 0.0) + means.get("scene_narration", 0.0)),
        ("Response serialization", means["response_json_serialization"]),
    ]
    candidates.sort(key=lambda item: item[1], reverse=True)

    final_adapter_ids = tuple(map(id, _ensure_live_adapters()))
    if final_adapter_ids != adapter_ids:
        raise RuntimeError("An inference adapter was replaced during measured frame processing")

    app.state.frame_profiler = None
    return {
        "conditions": {
            "input": "real local video frames encoded as JPEG/base64 and sent as browser frame messages",
            "video_path": str(video_path),
            **video_metadata,
            "frame_dimensions": [frames[0]["height"], frames[0]["width"]],
            "warmup_frames_excluded": warmup_frames,
            "measured_frames": measured_frames,
            "device": str(getattr(depth_adapter, "device", "unknown")),
            "depth_processor_input_size_configured": 364,
            "cached_adapters_initialized_before_frames": True,
            "adapter_initialization_ms_before_session": initialization_ms,
            "adapter_ids_unchanged_after_measurement": True,
            "websocket_transport": "FastAPI TestClient in-process ASGI WebSocket; not physical browser/network latency",
        },
        "stages": stages,
        "metadata": profile["metadata"],
        "comparisons": {
            "yolo_plus_bytetrack_mean_ms": detection_ms,
            "depth_anything_total_mean_ms": depth_ms,
            "backend_processing_total_mean_ms": total_backend_ms,
            "frame_to_response_round_trip_mean_ms": means["frame_to_response_round_trip"],
        },
        "derived": {
            "achievable_backend_fps": 1000.0 / total_backend_ms,
            "depth_percent_of_backend": depth_ms / total_backend_ms * 100,
            "detection_tracking_percent_of_backend": detection_ms / total_backend_ms * 100,
            "elsewhere_percent_of_backend": elsewhere_pct,
            "bottleneck_ranking": [
                {"stage": name, "mean_ms": value}
                for name, value in candidates
            ],
        },
        "query_path_note": (
            "Queries are separate WebSocket messages and do not run for each camera frame; "
            "scene description and spoken-scene-summary generation are timed per frame."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, default=DEFAULT_VIDEO)
    parser.add_argument("--frames", type=int, default=10, help="measured frames (minimum 10)")
    parser.add_argument("--warmup", type=int, default=2, help="unmeasured warmup frames")
    args = parser.parse_args()
    print(json.dumps(run_benchmark(args.video, measured_frames=args.frames, warmup_frames=args.warmup), indent=2))


if __name__ == "__main__":
    main()
