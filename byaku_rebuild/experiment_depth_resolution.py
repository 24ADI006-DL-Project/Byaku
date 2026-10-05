"""Controlled real-video experiment for Depth Anything V2 input-size policies."""
from __future__ import annotations

import argparse
import base64
from collections import Counter
import json
from pathlib import Path
import time
from typing import Any
from unittest.mock import patch

import cv2
import numpy as np
from fastapi.testclient import TestClient

from byaku_rebuild.api import app
from byaku_rebuild.detection_tracking import DEFAULT_WEIGHTS, DetectionTrackingAdapter
from byaku_rebuild.depth_integration import DEPTH_INPUT_SIZE, DepthAnythingV2Adapter
import byaku_rebuild.depth_integration as depth_module
from byaku_rebuild.frame_profiler import FrameProfiler
from byaku_rebuild.pipeline_orchestrator import PipelineOrchestrator


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_VIDEO = ROOT / "pinterest-video.mp4"
INPUT_POLICIES = (
    ("baseline_364", DEPTH_INPUT_SIZE),
    ("shorter_side_320", 320),
    ("shorter_side_256", 256),
    ("shorter_side_224", 224),
)
DEPTH_RELATION_THRESHOLD = 0.05
DEPTH_RELATIONS = {"nearer_than", "farther_than", "similar_depth"}
IMAGE_RELATIONS = {"left_of", "right_of", "above", "below", "overlapping"}


def load_benchmark_frames(
    video_path: Path,
    *,
    warmup_frames: int = 2,
    measured_frames: int = 10,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if warmup_frames < 1 or measured_frames < 1:
        raise ValueError("warmup_frames and measured_frames must be positive")
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open benchmark video: {video_path}")
    frames: list[dict[str, Any]] = []
    metadata = {
        "video_fps": float(capture.get(cv2.CAP_PROP_FPS)),
        "video_frame_count": int(capture.get(cv2.CAP_PROP_FRAME_COUNT)),
        "video_width": int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "video_height": int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
    }
    try:
        for index in range(warmup_frames + measured_frames):
            success, image = capture.read()
            if not success or image is None:
                raise RuntimeError(
                    f"Video ended after {index} frames; "
                    f"{warmup_frames + measured_frames} were required"
                )
            height, width = image.shape[:2]
            encoded, jpeg = cv2.imencode(".jpg", image)
            if not encoded:
                raise RuntimeError(f"JPEG encoding failed for source video frame {index}")
            frames.append(
                {
                    "source_index": index,
                    "image": image,
                    "width": width,
                    "height": height,
                    "data": base64.b64encode(jpeg.tobytes()).decode("ascii"),
                }
            )
    finally:
        capture.release()
    return frames, metadata


def _statistics(values: list[float]) -> dict[str, float | int]:
    if not values:
        raise ValueError("cannot summarize an empty measurement set")
    ordered = sorted(float(value) for value in values)
    middle = len(ordered) // 2
    median = (
        ordered[middle]
        if len(ordered) % 2
        else (ordered[middle - 1] + ordered[middle]) / 2
    )
    return {
        "count": len(ordered),
        "mean_ms": sum(ordered) / len(ordered),
        "median_ms": median,
        "min_ms": ordered[0],
        "max_ms": ordered[-1],
    }


def pairwise_depth_order_agreement(
    baseline_frames: list[dict[str, Any]],
    candidate_frames: list[dict[str, Any]],
    *,
    threshold: float = DEPTH_RELATION_THRESHOLD,
) -> dict[str, Any]:
    if len(baseline_frames) != len(candidate_frames):
        raise ValueError("baseline and candidate must contain the same number of frames")
    comparable = 0
    preserved = 0
    indeterminate = 0
    disagreements: list[dict[str, Any]] = []
    for frame_index, (baseline, candidate) in enumerate(zip(baseline_frames, candidate_frames)):
        base_objects = {
            str(item.get("object_id")): item
            for item in baseline.get("objects", [])
            if item.get("object_id") is not None
        }
        candidate_objects = {
            str(item.get("object_id")): item
            for item in candidate.get("objects", [])
            if item.get("object_id") is not None
        }
        shared_ids = sorted(base_objects.keys() & candidate_objects.keys())
        for index, first_id in enumerate(shared_ids):
            for second_id in shared_ids[index + 1:]:
                base_first = base_objects[first_id].get("relative_depth_value")
                base_second = base_objects[second_id].get("relative_depth_value")
                candidate_first = candidate_objects[first_id].get("relative_depth_value")
                candidate_second = candidate_objects[second_id].get("relative_depth_value")
                if any(
                    value is None or not np.isfinite(value)
                    for value in (base_first, base_second, candidate_first, candidate_second)
                ):
                    continue
                base_delta = float(base_first) - float(base_second)
                candidate_delta = float(candidate_first) - float(candidate_second)
                if abs(base_delta) <= threshold or abs(candidate_delta) <= threshold:
                    indeterminate += 1
                    continue
                comparable += 1
                is_preserved = (base_delta > 0) == (candidate_delta > 0)
                if is_preserved:
                    preserved += 1
                elif len(disagreements) < 12:
                    disagreements.append(
                        {
                            "frame_index": frame_index,
                            "first_object_id": first_id,
                            "second_object_id": second_id,
                            "baseline_relation": "first_nearer" if base_delta > 0 else "second_nearer",
                            "candidate_relation": (
                                "first_nearer" if candidate_delta > 0 else "second_nearer"
                            ),
                        }
                    )
    return {
        "comparable_pairs": comparable,
        "preserved_orders": preserved,
        "indeterminate_pairs": indeterminate,
        "agreement": preserved / comparable if comparable else None,
        "agreement_percent": preserved / comparable * 100 if comparable else None,
        "disagreements": disagreements,
    }


def object_depth_category_agreement(
    baseline_frames: list[dict[str, Any]],
    candidate_frames: list[dict[str, Any]],
) -> dict[str, Any]:
    if len(baseline_frames) != len(candidate_frames):
        raise ValueError("baseline and candidate must contain the same number of frames")
    comparable = 0
    preserved = 0
    disagreements: list[dict[str, Any]] = []
    for frame_index, (baseline, candidate) in enumerate(zip(baseline_frames, candidate_frames)):
        baseline_values = {
            str(item["object_id"]): float(item["relative_depth_value"])
            for item in baseline.get("objects", [])
            if item.get("object_id") is not None
            and item.get("relative_depth_value") is not None
            and np.isfinite(item["relative_depth_value"])
        }
        candidate_values = {
            str(item["object_id"]): float(item["relative_depth_value"])
            for item in candidate.get("objects", [])
            if item.get("object_id") is not None
            and item.get("relative_depth_value") is not None
            and np.isfinite(item["relative_depth_value"])
        }
        shared_ids = sorted(baseline_values.keys() & candidate_values.keys())
        if len(shared_ids) < 2:
            continue
        baseline_cutoffs = np.quantile(
            [baseline_values[object_id] for object_id in shared_ids],
            [1 / 3, 2 / 3],
        )
        candidate_cutoffs = np.quantile(
            [candidate_values[object_id] for object_id in shared_ids],
            [1 / 3, 2 / 3],
        )
        for object_id in shared_ids:
            base_depth = baseline_values[object_id]
            candidate_depth = candidate_values[object_id]
            base_category = (
                "farther"
                if base_depth < baseline_cutoffs[0]
                else "nearer"
                if base_depth > baseline_cutoffs[1]
                else "middle"
            )
            candidate_category = (
                "farther"
                if candidate_depth < candidate_cutoffs[0]
                else "nearer"
                if candidate_depth > candidate_cutoffs[1]
                else "middle"
            )
            comparable += 1
            if base_category == candidate_category:
                preserved += 1
            elif len(disagreements) < 12:
                disagreements.append(
                    {
                        "frame_index": frame_index,
                        "object_id": object_id,
                        "baseline_category": base_category,
                        "candidate_category": candidate_category,
                    }
                )
    return {
        "comparable_objects": comparable,
        "preserved_categories": preserved,
        "agreement_percent": preserved / comparable * 100 if comparable else None,
        "disagreements": disagreements,
    }


def _detection_signature(response: dict[str, Any]) -> list[tuple[Any, ...]]:
    signature = []
    for detection in response.get("detections", []):
        box = detection.get("bounding_box", detection.get("bbox", []))
        signature.append(
            (
                detection.get("track_id"),
                detection.get("class_name"),
                tuple(round(float(value), 3) for value in box),
            )
        )
    return sorted(signature, key=repr)


def _relation_signature(response: dict[str, Any], allowed: set[str]) -> set[tuple[str, str, str]]:
    return {
        (str(item.get("subject")), str(item.get("relation")), str(item.get("object")))
        for item in response.get("relationships", [])
        if item.get("relation") in allowed
    }


def _preview_check(response: dict[str, Any], width: int, height: int) -> dict[str, Any]:
    encoded_preview = response.get("depth_preview")
    if not isinstance(encoded_preview, str) or not encoded_preview:
        return {"valid": False, "reason": "depth_preview missing"}
    try:
        preview_bytes = base64.b64decode(encoded_preview, validate=True)
    except (ValueError, TypeError) as error:
        return {"valid": False, "reason": f"depth_preview is not valid base64: {error}"}
    preview = cv2.imdecode(np.frombuffer(preview_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
    if preview is None or preview.size == 0:
        return {"valid": False, "reason": "depth_preview JPEG did not decode"}
    preview_height, preview_width = preview.shape[:2]
    valid = (
        preview_width > 0
        and preview_height > 0
        and preview_width <= 640
        and preview_height <= 640
        and preview_width <= width
        and preview_height <= height
    )
    return {
        "valid": valid,
        "shape": [preview_height, preview_width],
        "reason": None if valid else "depth_preview dimensions are invalid",
    }


def _run_policy(
    *,
    name: str,
    requested_size: int,
    frames: list[dict[str, Any]],
    warmup_frames: int,
    profiler: FrameProfiler,
    adapter: DepthAnythingV2Adapter,
    results: list[dict[str, Any]],
    response_wall_times: list[float],
) -> dict[str, Any]:
    preview_checks: list[dict[str, Any]] = []
    if len(results) != len(frames) - warmup_frames:
        raise RuntimeError(f"{name} did not return every requested measured frame")
    for body, frame in zip(results, frames[warmup_frames:]):
        preview_checks.append(_preview_check(body, frame["width"], frame["height"]))
    profile = profiler.summary()
    stages = profile["stages"]
    expected_stage_counts = (
        "backend_processing_total",
        "depth_anything_total",
        "depth_preprocessing",
        "depth_model_inference",
        "depth_postprocessing",
        "yolo_bytetrack_total",
        "pipeline_orchestration_total",
        "spatial_relationships",
        "scene_memory",
        "scene_graph",
        "scene_description",
        "scene_narration",
        "response_payload_preparation",
    )
    for stage in expected_stage_counts:
        if stages.get(stage, {}).get("count") != len(results):
            raise RuntimeError(
                f"{name} stage {stage!r} measured "
                f"{stages.get(stage, {}).get('count', 0)} frames; expected {len(results)}"
            )

    metadata = profile["metadata"]
    depth_valid_frames = 0
    for body, frame in zip(results, frames[warmup_frames:]):
        if body.get("image_shape") != [frame["height"], frame["width"]]:
            raise RuntimeError(f"{name} response image dimensions differ from the source frame")
        object_depths_finite = all(
            item.get("relative_depth_value") is None
            or np.isfinite(item["relative_depth_value"])
            for item in body.get("objects", [])
        )
        if body.get("stage_status", {}).get("depth") == "ready" and object_depths_finite:
            depth_valid_frames += 1

    detections = [_detection_signature(body) for body in results]
    class_counts = Counter(
        str(signature[1])
        for frame_signature in detections
        for signature in frame_signature
    )
    representative_depths = [
        {
            "object_id": item.get("object_id"),
            "track_id": item.get("track_id"),
            "class_name": item.get("class_name"),
            "bounding_box": item.get("bounding_box"),
            "relative_depth_value": item.get("relative_depth_value"),
        }
        for item in results[0].get("objects", [])
    ]
    return {
        "name": name,
        "requested_size": requested_size,
        "results": results,
        "detection_signatures": detections,
        "detection_summary": {
            "detections_per_frame": [len(frame_signature) for frame_signature in detections],
            "total_detections": sum(len(frame_signature) for frame_signature in detections),
            "class_counts": dict(sorted(class_counts.items())),
            "track_ids_and_boxes_compared": True,
        },
        "representative_frame_depths": representative_depths,
        "stage_statistics": stages,
        "metadata": metadata,
        "backend_response_ms": _statistics(
            [float(body["processing_ms"]) for body in results]
        ),
        "http_round_trip_ms": _statistics(response_wall_times),
        "depth_preview": {
            "all_measured_frames_valid": all(check["valid"] for check in preview_checks),
            "representative_frame": preview_checks[0],
        },
        "depth_valid_frames": depth_valid_frames,
        "request_ids": [
            f"depth-experiment:{name}:{index + warmup_frames + 1}"
            for index in range(len(results))
        ],
        "model_load_count_during_frames": 0,
        "depth_adapter_identity": id(adapter),
    }


def _compare_image_relationships(
    baseline_frames: list[dict[str, Any]],
    candidate_frames: list[dict[str, Any]],
) -> dict[str, Any]:
    identical = 0
    total = len(baseline_frames)
    disagreements: list[dict[str, Any]] = []
    for index, (baseline, candidate) in enumerate(zip(baseline_frames, candidate_frames)):
        base_relations = _relation_signature(baseline, IMAGE_RELATIONS)
        candidate_relations = _relation_signature(candidate, IMAGE_RELATIONS)
        if base_relations == candidate_relations:
            identical += 1
        elif len(disagreements) < 5:
            disagreements.append(
                {
                    "frame_index": index,
                    "baseline_only": sorted(base_relations - candidate_relations)[:8],
                    "candidate_only": sorted(candidate_relations - base_relations)[:8],
                }
            )
    return {
        "identical_frames": identical,
        "measured_frames": total,
        "agreement_percent": identical / total * 100 if total else None,
        "disagreements": disagreements,
    }


def _compare_depth_relationships(
    baseline_frames: list[dict[str, Any]],
    candidate_frames: list[dict[str, Any]],
) -> dict[str, Any]:
    baseline_counts = Counter(
        item.get("relation")
        for response in baseline_frames
        for item in response.get("relationships", [])
        if item.get("relation") in DEPTH_RELATIONS
    )
    candidate_counts = Counter(
        item.get("relation")
        for response in candidate_frames
        for item in response.get("relationships", [])
        if item.get("relation") in DEPTH_RELATIONS
    )
    disagreements: list[dict[str, Any]] = []
    differing_frames = 0
    for index, (baseline, candidate) in enumerate(zip(baseline_frames, candidate_frames)):
        base = _relation_signature(baseline, DEPTH_RELATIONS)
        other = _relation_signature(candidate, DEPTH_RELATIONS)
        if base != other:
            differing_frames += 1
            if len(disagreements) < 12:
                disagreements.extend(
                    {
                        "frame_index": index,
                        "baseline": relation,
                        "candidate": None,
                    }
                    for relation in sorted(base - other)[:6]
                )
                disagreements.extend(
                    {
                        "frame_index": index,
                        "baseline": None,
                        "candidate": relation,
                    }
                    for relation in sorted(other - base)[:6]
                )
    return {
        "baseline_relation_counts": dict(sorted(baseline_counts.items())),
        "candidate_relation_counts": dict(sorted(candidate_counts.items())),
        "frames_with_relation_changes": differing_frames,
        "disagreements": disagreements[:12],
    }


def run_experiment(
    video_path: Path = DEFAULT_VIDEO,
    *,
    measured_frames: int = 10,
    warmup_frames: int = 2,
) -> dict[str, Any]:
    if measured_frames != 10 or warmup_frames != 2:
        raise ValueError("the controlled experiment requires exactly 2 warm-up and 10 measured frames")
    video_path = video_path.resolve()
    if not video_path.is_file():
        raise FileNotFoundError(f"Local video not found: {video_path}")
    frames, video_metadata = load_benchmark_frames(
        video_path,
        warmup_frames=warmup_frames,
        measured_frames=measured_frames,
    )
    app.state.depth_adapter = DepthAnythingV2Adapter(device="cpu")
    depth_adapter = app.state.depth_adapter
    if str(depth_adapter.device) != "cpu":
        raise RuntimeError(f"Experiment requires CPU; selected {depth_adapter.device}")
    policy_names = [name for name, _size in INPUT_POLICIES]
    detection_adapters = []
    detection_adapter_initialization_ms = []
    for _ in policy_names:
        init_started = time.perf_counter()
        detection_adapters.append(DetectionTrackingAdapter(weights_path=DEFAULT_WEIGHTS))
        detection_adapter_initialization_ms.append((time.perf_counter() - init_started) * 1000)
    detection_adapter_identities = tuple(id(adapter) for adapter in detection_adapters)
    adapter_identity = id(depth_adapter)
    profiles = {name: FrameProfiler() for name in policy_names}
    orchestrators = {name: PipelineOrchestrator() for name in policy_names}
    bodies_by_policy: dict[str, list[dict[str, Any]]] = {
        name: [] for name in policy_names
    }
    round_trip_by_policy: dict[str, list[float]] = {
        name: [] for name in policy_names
    }

    def submit_frame(client: TestClient, name: str, frame_index: int) -> tuple[dict[str, Any], float]:
        policy_index = next(
            index for index, (policy_name, _size) in enumerate(INPUT_POLICIES)
            if policy_name == name
        )
        requested_size = INPUT_POLICIES[policy_index][1]
        app.state.orchestrator = orchestrators[name]
        app.state.detection_adapter = detection_adapters[policy_index]
        app.state.depth_adapter = depth_adapter
        app.state.frame_profiler = profiles[name]
        frame = frames[frame_index]
        request = {
            "frame_id": f"depth-experiment:{name}:{frame_index + 1}",
            "frame_number": frame_index + 1,
            "width": frame["width"],
            "height": frame["height"],
            "data": frame["data"],
        }
        started = time.perf_counter()
        with patch.object(depth_module, "DEPTH_INPUT_SIZE", requested_size):
            response = client.post("/api/frame", json=request)
        round_trip_ms = (time.perf_counter() - started) * 1000
        if response.status_code != 200:
            raise RuntimeError(
                f"{name} frame {frame_index + 1} returned HTTP "
                f"{response.status_code}: {response.text}"
            )
        body = response.json()
        if body.get("status") not in {"success", "partial"}:
            raise RuntimeError(f"{name} frame {frame_index + 1} failed: {body!r}")
        return body, round_trip_ms

    with TestClient(app) as client:
        for frame_index in range(warmup_frames):
            for name in policy_names:
                submit_frame(client, name, frame_index)
        for profiler in profiles.values():
            profiler.clear()
        policy_count = len(INPUT_POLICIES)
        for frame_index in range(warmup_frames, len(frames)):
            first_policy = (frame_index - warmup_frames) % policy_count
            for offset in range(policy_count):
                name = INPUT_POLICIES[(first_policy + offset) % policy_count][0]
                body, round_trip_ms = submit_frame(client, name, frame_index)
                bodies_by_policy[name].append(body)
                round_trip_by_policy[name].append(round_trip_ms)

    policy_results = []
    for policy_index, (name, requested_size) in enumerate(INPUT_POLICIES):
        result = _run_policy(
            name=name,
            requested_size=requested_size,
            frames=frames,
            warmup_frames=warmup_frames,
            profiler=profiles[name],
            adapter=depth_adapter,
            results=bodies_by_policy[name],
            response_wall_times=round_trip_by_policy[name],
        )
        result["detection_adapter_initialization_ms"] = detection_adapter_initialization_ms[policy_index]
        policy_results.append(result)

    baseline = policy_results[0]
    baseline_detections = baseline["detection_signatures"]
    comparisons = {}
    for candidate in policy_results[1:]:
        detection_comparison = {
            "identical_frames": sum(
                left == right
                for left, right in zip(baseline_detections, candidate["detection_signatures"])
            ),
            "measured_frames": measured_frames,
            "identical": baseline_detections == candidate["detection_signatures"],
            "first_mismatch": next(
                (
                    index
                    for index, (left, right) in enumerate(
                        zip(baseline_detections, candidate["detection_signatures"])
                    )
                    if left != right
                ),
                None,
            ),
        }
        comparisons[candidate["name"]] = {
            "depth_order": pairwise_depth_order_agreement(
                baseline["results"],
                candidate["results"],
            ),
            "object_depth_categories": object_depth_category_agreement(
                baseline["results"],
                candidate["results"],
            ),
            "image_space_relationships": _compare_image_relationships(
                baseline["results"],
                candidate["results"],
            ),
            "depth_relationships": _compare_depth_relationships(
                baseline["results"],
                candidate["results"],
            ),
            "detection_tracking": detection_comparison,
        }

    output_policies = []
    for result in policy_results:
        stages = result["stage_statistics"]
        backend_ms = stages["backend_processing_total"]["mean_ms"]
        depth_ms = stages["depth_anything_total"]["mean_ms"]
        detection_ms = stages["yolo_bytetrack_total"]["mean_ms"]
        output_policies.append(
            {
                "name": result["name"],
                "requested_size": result["requested_size"],
                "actual_tensor_shape": result["metadata"].get("depth_model_input_tensor_shape"),
                "processor_source_image_shape": result["metadata"].get(
                    "depth_processor_source_image_shape"
                ),
                "processor_size_argument": result["metadata"].get(
                    "depth_processor_size_argument"
                ),
                "processor_keep_aspect_ratio": result["metadata"].get(
                    "depth_processor_keep_aspect_ratio"
                ),
                "processor_ensure_multiple_of": result["metadata"].get(
                    "depth_processor_ensure_multiple_of"
                ),
                "processor_class": result["metadata"].get("depth_processor_class"),
                "processor_do_resize": result["metadata"].get("depth_processor_do_resize"),
                "processor_do_pad": result["metadata"].get("depth_processor_do_pad"),
                "processor_crop_size": result["metadata"].get("depth_processor_crop_size"),
                "model_output_tensor_shape": result["metadata"].get(
                    "depth_model_output_tensor_shape"
                ),
                "depth_ms": stages["depth_anything_total"],
                "depth_preprocessing_ms": stages["depth_preprocessing"],
                "depth_model_inference_ms": stages["depth_model_inference"],
                "depth_postprocessing_ms": stages["depth_postprocessing"],
                "yolo_bytetrack_ms": stages["yolo_bytetrack_total"],
                "spatial_relationships_ms": stages["spatial_relationships"],
                "scene_memory_ms": stages["scene_memory"],
                "scene_graph_ms": stages["scene_graph"],
                "scene_description_ms": stages["scene_description"],
                "scene_narration_ms": stages["scene_narration"],
                "pipeline_orchestration_ms": stages["pipeline_orchestration_total"],
                "response_preparation_ms": stages["response_payload_preparation"],
                "backend_processing_ms": stages["backend_processing_total"],
                "http_round_trip_ms": result["http_round_trip_ms"],
                "other_ms": backend_ms - depth_ms - detection_ms,
                "fps": 1000 / backend_ms,
                "depth_valid_on_all_frames": result["depth_valid_frames"] == measured_frames,
                "depth_valid_frame_count": result["depth_valid_frames"],
                "depth_preview": result["depth_preview"],
                "detection_summary": result["detection_summary"],
                "representative_frame_depths": result["representative_frame_depths"],
                "detection_signatures": result["detection_signatures"],
                "results": result["results"],
            }
        )

    if (
        adapter_identity != id(depth_adapter)
        or detection_adapter_identities != tuple(id(adapter) for adapter in detection_adapters)
    ):
        raise RuntimeError("A model adapter identity changed during the experiment")
    for candidate in output_policies:
        candidate.pop("detection_signatures")
        candidate.pop("results")
    return {
        "conditions": {
            "video_path": str(video_path),
            **video_metadata,
            "source_frames": [item["source_index"] for item in frames],
            "warmup_frames": warmup_frames,
            "measured_frames": measured_frames,
            "device": "cpu",
            "policies": [
                {"name": name, "requested_size": size}
                for name, size in INPUT_POLICIES
            ],
            "same_video_frames_for_all_policies": True,
            "depth_model_loaded_once_and_reused": True,
            "model_adapters_created_before_frame_processing": True,
            "model_adapters_unchanged_during_measured_frames": True,
            "first_call_predictor_warmup_excluded": True,
            "detection_tracker_independent_per_policy": True,
            "detection_same_frame_sequence_per_policy": True,
            "detection_adapter_initialization_ms": {
                result["name"]: result["detection_adapter_initialization_ms"]
                for result in policy_results
            },
            "processor_size_was_temporary_runtime_override_only": True,
            "baseline_configuration_unchanged": True,
        },
        "policies": output_policies,
        "comparisons_to_baseline": comparisons,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, default=DEFAULT_VIDEO)
    args = parser.parse_args()
    print(json.dumps(run_experiment(args.video), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
