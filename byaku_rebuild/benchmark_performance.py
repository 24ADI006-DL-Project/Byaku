"""BYAKU Step 24: Benchmark script to establish performance baselines.

Measures:
- YOLO detection & ByteTrack latency
- Depth Anything V2 (pre-processing, inference, post-processing)
- Pipeline orchestrator (spatial, scene graph, memory, description)
- Full frame processing latency (steady-state vs warm-up)
- Query response latency
"""
import base64
import json
from pathlib import Path
import time
from typing import Any

import cv2
import numpy as np
import torch

from byaku_rebuild.api import FrameRequest, _process_live_frame_unlocked, _ensure_live_adapters, app
from byaku_rebuild.query_reasoning import answer_query


ROOT = Path(__file__).resolve().parent.parent
TEST_IMG_1 = ROOT / "test1.jpg"
TEST_IMG_2 = ROOT / "test2.jpg"


def load_image_b64(path: Path) -> tuple[str, int, int, np.ndarray]:
    frame = cv2.imread(str(path))
    if frame is None:
        raise FileNotFoundError(f"Could not load image: {path}")
    h, w = frame.shape[:2]
    _, buffer = cv2.imencode(".jpg", frame)
    b64_str = base64.b64encode(buffer).decode("ascii")
    return b64_str, w, h, frame


def benchmark_component_breakdown(iterations: int = 5) -> dict[str, Any]:
    b64_1, w1, h1, frame1 = load_image_b64(TEST_IMG_1)
    detection_adapter, depth_adapter, orchestrator = _ensure_live_adapters()

    # Warm-up run
    print("Warming up models...")
    _ = depth_adapter.estimate_depth(frame1)
    _ = detection_adapter.process_frame(frame1, frame_id="warmup-1", frame_number=90001)

    print(f"\nRunning {iterations} iterations on {TEST_IMG_1.name} ({w1}x{h1})...")

    yolo_times = []
    depth_prep_times = []
    depth_infer_times = []
    depth_post_times = []
    depth_total_times = []
    orchestrator_times = []
    full_pipeline_times = []

    base_num = 100000
    for i in range(iterations):
        # 1. Depth estimation breakdown
        t_d0 = time.perf_counter()
        rgb_frame = cv2.cvtColor(frame1, cv2.COLOR_BGR2RGB)
        from PIL import Image
        processor_image = Image.fromarray(rgb_frame)
        inputs = depth_adapter.processor(images=processor_image, return_tensors="pt")
        inputs = {k: v.to(depth_adapter.device) for k, v in inputs.items()}
        t_d1 = time.perf_counter()

        with torch.inference_mode():
            output = depth_adapter.model(**inputs)
        t_d2 = time.perf_counter()

        raw_pred = output.predicted_depth
        from byaku_rebuild.depth_integration import _to_numpy, normalize_relative_depth_map
        raw_array = _to_numpy(raw_pred)
        norm_result = normalize_relative_depth_map(raw_array, (h1, w1))
        t_d3 = time.perf_counter()

        depth_prep_times.append((t_d1 - t_d0) * 1000)
        depth_infer_times.append((t_d2 - t_d1) * 1000)
        depth_post_times.append((t_d3 - t_d2) * 1000)
        depth_total_times.append((t_d3 - t_d0) * 1000)

        # 2. YOLO + ByteTrack
        frame_num_det = base_num + (i * 2) + 1
        frame_id_det = f"bench:{frame_num_det}"
        t_y0 = time.perf_counter()
        det_result = detection_adapter.process_frame(frame1, frame_id=frame_id_det, frame_number=frame_num_det)
        t_y1 = time.perf_counter()
        yolo_times.append((t_y1 - t_y0) * 1000)

        # 3. Orchestrator
        t_o0 = time.perf_counter()
        pipe_result = orchestrator.process_frame(
            frame_id=frame_id_det,
            frame_number=frame_num_det,
            image_dimensions={"width": w1, "height": h1},
            detections=det_result.get("orchestrator_detections", []),
            depth_map=norm_result.get("depth_map"),
            timestamp=time.time(),
        )
        t_o1 = time.perf_counter()
        orchestrator_times.append((t_o1 - t_o0) * 1000)

        # 4. Total API unlocked pipeline call
        frame_num_api = base_num + (i * 2) + 2
        frame_id_api = f"bench-api:{frame_num_api}"
        req = FrameRequest(
            frame_id=frame_id_api,
            frame_number=frame_num_api,
            width=w1,
            height=h1,
            data=b64_1,
            timestamp=time.time(),
        )
        t_f0 = time.perf_counter()
        api_result = _process_live_frame_unlocked(req)
        t_f1 = time.perf_counter()
        full_pipeline_times.append((t_f1 - t_f0) * 1000)

    # 5. Query latency
    query_times = []
    for q in ["Where is the bed?", "What is to the left of the bed?", "Is a chair visible?"]:
        t_q0 = time.perf_counter()
        _ = orchestrator.query(q)
        t_q1 = time.perf_counter()
        query_times.append((t_q1 - t_q0) * 1000)

    report = {
        "yolo_bytetrack_ms": {
            "mean": float(np.mean(yolo_times)),
            "min": float(np.min(yolo_times)),
            "max": float(np.max(yolo_times)),
        },
        "depth_anything_breakdown_ms": {
            "prep_mean": float(np.mean(depth_prep_times)),
            "infer_mean": float(np.mean(depth_infer_times)),
            "post_mean": float(np.mean(depth_post_times)),
            "total_mean": float(np.mean(depth_total_times)),
            "total_min": float(np.min(depth_total_times)),
            "total_max": float(np.max(depth_total_times)),
        },
        "orchestrator_ms": {
            "mean": float(np.mean(orchestrator_times)),
            "min": float(np.min(orchestrator_times)),
            "max": float(np.max(orchestrator_times)),
        },
        "full_frame_processing_ms": {
            "mean": float(np.mean(full_pipeline_times)),
            "min": float(np.min(full_pipeline_times)),
            "max": float(np.max(full_pipeline_times)),
            "effective_fps": float(1000.0 / np.mean(full_pipeline_times)),
        },
        "query_latency_ms": {
            "mean": float(np.mean(query_times)),
            "min": float(np.min(query_times)),
            "max": float(np.max(query_times)),
        },
    }
    return report


if __name__ == "__main__":
    report = benchmark_component_breakdown(iterations=5)
    print("\n=== BASELINE PERFORMANCE REPORT ===")
    print(json.dumps(report, indent=2))
