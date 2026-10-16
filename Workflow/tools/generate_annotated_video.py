"""Generate annotated video with object bbox tracking, depth, and voice narration."""

from __future__ import annotations

from pathlib import Path
import json
import time
from typing import Any

import cv2
import numpy as np
import pyttsx3

from Workflow.vision.object_detector_tracker import DetectionTrackingAdapter
from Workflow.vision.relative_depth_estimator import DepthAnythingV2Adapter
from Workflow.scene_pipeline import PipelineOrchestrator


DEPTH_COLORS_BGR = {
    "NEAR": (0, 0, 255),       # Red
    "MID-RANGE": (0, 220, 255), # Yellow-Gold
    "FAR": (255, 140, 0),      # Blue-Cyan
    "UNKNOWN": (140, 140, 140), # Gray
}


def draw_annotated_frame(
    frame: np.ndarray,
    objects: list[dict[str, Any]],
    current_narration: str,
    frame_number: int,
    total_frames: int,
    fps: float,
) -> np.ndarray:
    annotated = frame.copy()
    height, width = annotated.shape[:2]

    # 1. Draw detected objects with persistent tracking and depth
    for item in objects:
        box = item.get("bounding_box")
        if not isinstance(box, (list, tuple)) or len(box) != 4:
            continue
        try:
            x1, y1, x2, y2 = (round(float(v)) for v in box)
        except (TypeError, ValueError, OverflowError):
            continue
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(width - 1, x2), min(height - 1, y2)
        if x2 <= x1 or y2 <= y1:
            continue

        category = str(item.get("depth_category") or "UNKNOWN").upper()
        color = DEPTH_COLORS_BGR.get(category, DEPTH_COLORS_BGR["UNKNOWN"])
        track_id = item.get("track_id")
        track_label = f"ID #{track_id}" if track_id is not None else "NEW"
        class_name = str(item.get("class_name") or "object").upper()
        confidence = item.get("confidence")
        conf_str = f" {float(confidence):.0%}" if confidence is not None else ""
        position = str(item.get("horizontal_position") or "CENTER").upper()

        # Bounding box
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 3)

        # Corner accents for sleek modern HUD look
        corner_len = min(20, (x2 - x1) // 4, (y2 - y1) // 4)
        if corner_len > 4:
            # Top-left
            cv2.line(annotated, (x1, y1), (x1 + corner_len, y1), (255, 255, 255), 2)
            cv2.line(annotated, (x1, y1), (x1, y1 + corner_len), (255, 255, 255), 2)
            # Bottom-right
            cv2.line(annotated, (x2, y2), (x2 - corner_len, y2), (255, 255, 255), 2)
            cv2.line(annotated, (x2, y2), (x2, y2 - corner_len), (255, 255, 255), 2)

        # Label pill
        label = f"{class_name}{conf_str} | {track_label} | {position} | {category}"
        font_scale = 0.52
        thickness = 1
        (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
        label_y2 = max(th + baseline + 6, y1)
        label_y1 = label_y2 - th - baseline - 6
        label_x2 = min(width - 1, x1 + tw + 10)

        # Draw label background
        cv2.rectangle(annotated, (x1, label_y1), (label_x2, label_y2), color, cv2.FILLED)
        cv2.rectangle(annotated, (x1, label_y1), (label_x2, label_y2), (0, 0, 0), 1)

        # Label text (white on dark color, black on light color)
        text_color = (0, 0, 0) if category == "MID-RANGE" else (255, 255, 255)
        cv2.putText(
            annotated,
            label,
            (x1 + 5, label_y2 - baseline - 3),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            text_color,
            thickness,
            cv2.LINE_AA,
        )

    # 2. Header HUD banner
    hud_height = 44
    overlay = annotated.copy()
    cv2.rectangle(overlay, (0, 0), (width, hud_height), (15, 15, 20), cv2.FILLED)
    cv2.addWeighted(overlay, 0.85, annotated, 0.15, 0, annotated)

    hud_text = f"BYAKU VISION ASSISTANT  |  TRACKING ACTIVE  |  FRAME {frame_number}/{total_frames}  |  OBJECTS: {len(objects)}"
    cv2.putText(annotated, hud_text, (16, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (0, 240, 255), 1, cv2.LINE_AA)

    # 3. Bottom Assistive Telemetry & Narration Subtitle Banner
    sub_height = 90
    sub_y1 = height - sub_height
    overlay_sub = annotated.copy()
    cv2.rectangle(overlay_sub, (0, sub_y1), (width, height), (10, 10, 15), cv2.FILLED)
    cv2.addWeighted(overlay_sub, 0.88, annotated, 0.12, 0, annotated)

    # Accent line separating video and subtitle banner
    cv2.line(annotated, (0, sub_y1), (width, sub_y1), (0, 240, 255), 2)

    # Speaker title
    speaker_title = "ASSISTIVE SCENE NARRATION:"
    cv2.putText(annotated, speaker_title, (16, sub_y1 + 24), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 255, 200), 1, cv2.LINE_AA)

    # Narration text (wrapped if necessary)
    if current_narration:
        words = current_narration.split()
        lines = []
        cur_line = ""
        for w in words:
            test_line = f"{cur_line} {w}".strip()
            (lw, _), _ = cv2.getTextSize(test_line, cv2.FONT_HERSHEY_SIMPLEX, 0.52, 1)
            if lw < width - 40:
                cur_line = test_line
            else:
                lines.append(cur_line)
                cur_line = w
        if cur_line:
            lines.append(cur_line)

        for i, line in enumerate(lines[:2]):
            cv2.putText(
                annotated,
                line,
                (16, sub_y1 + 48 + i * 22),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.52,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

    return annotated


def process_video_and_annotate(
    video_path: Path,
    output_video_path: Path,
    output_audio_path: Path,
) -> dict[str, Any]:
    print(f"[1/4] Opening source video: {video_path}")
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video file: {video_path}")

    fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration = total_frames / fps
    print(f"      Properties: {width}x{height} @ {fps:.1f} FPS, {total_frames} frames ({duration:.2f}s)")

    print("[2/4] Initializing BYAKU AI Vision Pipeline (YOLO11n + ByteTrack + Depth Anything V2)...")
    detector = DetectionTrackingAdapter(imgsz=640)
    depth_estimator = DepthAnythingV2Adapter()
    orchestrator = PipelineOrchestrator()

    # Video writer
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    output_video_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(output_video_path), fourcc, fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Could not create video writer for: {output_video_path}")

    frame_index = 0
    narration_history = []
    current_narration = "Analyzing indoor scene..."
    last_depth_map = None
    processed_objects_per_frame = []

    print("[3/4] Processing frames and rendering annotations...")
    t_start = time.perf_counter()

    while True:
        success, frame = capture.read()
        if not success or frame is None:
            break
        frame_index += 1

        # Run detection and tracking on every frame
        det_result = detector.process_frame(
            frame,
            frame_id=f"video:{frame_index}",
            frame_number=frame_index,
        )
        detections = det_result.get("detections", [])

        # Compute depth every 6 frames (5 times per second at 30fps) for smooth tracking speed
        if frame_index % 6 == 1 or last_depth_map is None:
            depth_result = depth_estimator.estimate_depth(frame)
            last_depth_map = depth_result.get("depth_map")

        # Run scene reasoning orchestrator
        pipeline_res = orchestrator.process_frame(
            detections=detections,
            depth_map=last_depth_map,
            frame_id=f"video:{frame_index}",
            frame_number=frame_index,
            image_dimensions=(height, width),
        )

        objects = pipeline_res.get("objects", [])
        processed_objects_per_frame.append(objects)

        # Update scene narration when updated
        desc_info = pipeline_res.get("scene_description", {})
        speech_info = pipeline_res.get("scene_summary_speech", {})
        new_speech = speech_info.get("spoken_text") if speech_info else None
        if not new_speech and desc_info:
            new_speech = desc_info.get("description")
        if new_speech and new_speech != current_narration:
            current_narration = new_speech
            narration_history.append({"frame": frame_index, "timestamp_s": frame_index / fps, "text": new_speech})

        # Draw annotations and telemetry
        annotated_frame = draw_annotated_frame(
            frame,
            objects,
            current_narration,
            frame_index,
            total_frames,
            fps,
        )

        writer.write(annotated_frame)
        if frame_index % 25 == 0 or frame_index == total_frames:
            elapsed = time.perf_counter() - t_start
            rate = frame_index / elapsed if elapsed > 0 else 0
            print(f"      Rendered frame {frame_index}/{total_frames} ({frame_index/total_frames:.0%}) at {rate:.1f} fps...")

    capture.release()
    writer.release()
    total_elapsed = time.perf_counter() - t_start
    print(f"      Render complete: {output_video_path.name} in {total_elapsed:.1f}s ({output_video_path.stat().st_size / (1024*1024):.2f} MB)")

    # 4. Generate synchronized voice narration
    print("[4/4] Synthesizing speech narration audio with pyttsx3...")
    engine = pyttsx3.init()
    engine.setProperty("rate", 160)  # comfortable, intelligible speech rate

    # Collect unique narration texts or comprehensive summary
    full_narration_texts = []
    if narration_history:
        for entry in narration_history:
            txt = entry["text"].strip()
            if txt and (not full_narration_texts or txt != full_narration_texts[-1]):
                full_narration_texts.append(txt)
    else:
        full_narration_texts.append("Indoor scene analyzed. Objects detected and tracked.")

    final_narration_script = " ".join(full_narration_texts)
    print(f"      Narration Script: \"{final_narration_script}\"")

    output_audio_path.parent.mkdir(parents=True, exist_ok=True)
    engine.save_to_file(final_narration_script, str(output_audio_path))
    engine.runAndWait()

    # Save metadata JSON for the player
    metadata_path = output_video_path.with_suffix(".json")
    metadata = {
        "source_video": str(video_path),
        "annotated_video": output_video_path.name,
        "narration_audio": output_audio_path.name,
        "width": width,
        "height": height,
        "fps": fps,
        "total_frames": total_frames,
        "duration_seconds": duration,
        "narration_script": final_narration_script,
        "narration_cues": narration_history,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"      Saved metadata to: {metadata_path.name}")

    return metadata


if __name__ == "__main__":
    src = Path("Data/samples/indoor_objects_video.mp4").resolve()
    out_video = Path("Data/samples/indoor_objects_video_annotated.mp4").resolve()
    out_audio = Path("Data/samples/indoor_objects_video_narration.wav").resolve()
    result = process_video_and_annotate(src, out_video, out_audio)
    print("\nSUCCESS! Video annotation and narration generation complete.")
