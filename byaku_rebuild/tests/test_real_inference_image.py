"""
Step 22 -- Real AI Inference Single-Image Test.

Loads local YOLO11n weights and Depth Anything V2 Small from local cache,
runs real inference on test1.jpg, passes outputs through the rebuild
spatial/scene-graph/memory pipeline, and saves visual artifacts into results/.
"""
from __future__ import annotations

from pathlib import Path
import sys
import time

WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

import cv2
import numpy as np
import pytest

from byaku_rebuild.detection_tracking import DEFAULT_WEIGHTS, DetectionTrackingAdapter
from byaku_rebuild.depth_integration import MODEL_NAME, DepthAnythingV2Adapter
from byaku_rebuild.pipeline_orchestrator import PipelineOrchestrator


WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"


def run_single_image_inference(image_name: str = "test1.jpg") -> dict:
    image_path = WORKSPACE_ROOT / image_name
    assert image_path.is_file(), f"Test image not found: {image_path}"
    image = cv2.imread(str(image_path))
    assert image is not None, f"Failed to load image: {image_path}"
    height, width = image.shape[:2]

    # 1. Real YOLO Detection & ByteTrack
    t0 = time.perf_counter()
    detection_adapter = DetectionTrackingAdapter(weights_path=DEFAULT_WEIGHTS)
    detection_result = detection_adapter.process_frame(
        image,
        frame_id="img-test-1",
        frame_number=1,
    )
    det_time_ms = (time.perf_counter() - t0) * 1000.0

    # 2. Real Depth Anything V2 Inference
    t0 = time.perf_counter()
    depth_adapter = DepthAnythingV2Adapter(model_name=MODEL_NAME, device="cpu")
    depth_result = depth_adapter.estimate_depth(image)
    depth_time_ms = (time.perf_counter() - t0) * 1000.0

    # 3. Validation
    assert detection_result["status"] == "success"
    assert depth_result["status"] == "success"
    assert depth_result["depth_available"] is True
    assert depth_result["depth_map"].shape == (height, width)
    assert depth_result["metadata"]["metric_distance_available"] is False

    # 4. Pipeline Orchestration
    orchestrator = PipelineOrchestrator()
    t0 = time.perf_counter()
    pipeline_result = orchestrator.process_frame(
        frame_id="img-test-1",
        frame_number=1,
        image_dimensions={"width": width, "height": height},
        detections=detection_result.get("orchestrator_detections", []),
        depth_map=depth_result["depth_map"],
    )
    orch_time_ms = (time.perf_counter() - t0) * 1000.0

    # 5. Save visualizations under byaku_rebuild/results/
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Detections visualization
    vis_image = image.copy()
    for det in detection_result.get("detections", []):
        box = [int(v) for v in det["bounding_box"]]
        cls_name = det["class_name"]
        conf = det["confidence"]
        cv2.rectangle(vis_image, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
        label = f"{cls_name} {conf:.2f}"
        cv2.putText(vis_image, label, (box[0], max(box[1] - 5, 15)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    det_vis_path = RESULTS_DIR / "real_inference_detections.png"
    cv2.imwrite(str(det_vis_path), vis_image)

    # Depth map visualization
    depth_norm = (depth_result["depth_map"] * 255).astype(np.uint8)
    depth_vis = cv2.applyColorMap(depth_norm, cv2.COLORMAP_INFERNO)
    depth_vis_path = RESULTS_DIR / "real_inference_depth.png"
    cv2.imwrite(str(depth_vis_path), depth_vis)

    return {
        "image_shape": (height, width),
        "detections": detection_result.get("detections", []),
        "detection_time_ms": det_time_ms,
        "depth_available": depth_result["depth_available"],
        "depth_shape": depth_result["depth_map"].shape,
        "depth_time_ms": depth_time_ms,
        "orchestrator_time_ms": orch_time_ms,
        "pipeline_result": pipeline_result,
        "detection_vis": str(det_vis_path),
        "depth_vis": str(depth_vis_path),
    }


def test_real_ai_inference_on_local_image() -> None:
    res = run_single_image_inference("test1.jpg")

    assert len(res["detections"]) > 0, "Expected at least one detection on test1.jpg"
    assert res["depth_available"] is True
    assert res["depth_shape"] == res["image_shape"]

    # Verify no metric distance
    for obj in res["pipeline_result"].get("objects", []):
        assert "distance_meters" not in obj
        assert "distance_m" not in obj

    desc = res["pipeline_result"].get("scene_description", {})
    assert desc.get("visible_object_count", 0) > 0


if __name__ == "__main__":
    out = run_single_image_inference("test1.jpg")
    print(f"Image Shape: {out['image_shape']}")
    print(f"Detection Latency: {out['detection_time_ms']:.1f}ms")
    print(f"Detections Count: {len(out['detections'])}")
    for d in out["detections"]:
        print(f"  - {d['class_name']}: conf={d['confidence']:.3f}, box={d['bounding_box']}")
    print(f"Depth Latency: {out['depth_time_ms']:.1f}ms")
    print(f"Depth Available: {out['depth_available']}, Shape: {out['depth_shape']}")
    print(f"Orchestration Latency: {out['orchestrator_time_ms']:.1f}ms")
    desc = out["pipeline_result"].get("scene_description", {})
    print(f"Scene Description: {desc.get('description')}")
    summary = out["pipeline_result"].get("scene_summary_speech", {})
    print(f"Summary Speech: {summary.get('spoken_text')}")
    print(f"Visualizations saved to:\n  {out['detection_vis']}\n  {out['depth_vis']}")
