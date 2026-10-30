from __future__ import annotations

import base64
from typing import Any

try:
    # pyrefly: ignore [missing-import]
    import cv2
except Exception:  # pragma: no cover
    cv2 = None  # type: ignore

try:
    # pyrefly: ignore [missing-import]
    import numpy as np
except Exception:  # pragma: no cover
    np = None  # type: ignore


DEPTH_COLORS_BGR = {
    "NEAR": (0, 0, 255),
    "MID-RANGE": (0, 255, 255),
    "FAR": (255, 0, 0),
}


def encode_annotated_frame(frame: Any, objects: list[dict[str, Any]]) -> tuple[str, tuple[int, int]]:
    """Encode a bounded preview annotated from this frame's own scene objects."""
    if frame is None or not hasattr(frame, "shape"):
        raise ValueError("invalid frame provided for annotation")
    if cv2 is None:
        raise RuntimeError("OpenCV is required to encode annotated frame")

    original_height, original_width = frame.shape[:2]
    scale = min(1.0, 1280 / max(original_width, original_height))
    if scale < 1:
        annotated = cv2.resize(
            frame,
            (round(original_width * scale), round(original_height * scale)),
            interpolation=cv2.INTER_AREA,
        )
    else:
        annotated = frame.copy()
    preview_height, preview_width = annotated.shape[:2]

    for item in objects:
        box = item.get("bounding_box")
        if not isinstance(box, (list, tuple)) or len(box) != 4:
            continue
        try:
            x1, y1, x2, y2 = (round(float(value) * scale) for value in box)
        except (TypeError, ValueError, OverflowError):
            continue
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(preview_width - 1, x2), min(preview_height - 1, y2)
        if x2 <= x1 or y2 <= y1:
            continue

        category = item.get("depth_category")
        if category not in DEPTH_COLORS_BGR:
            category = "UNKNOWN"
        color = DEPTH_COLORS_BGR.get(category, (128, 128, 128))
        track_id = item.get("track_id")
        track_label = f"ID {track_id}" if track_id is not None else "ID unavailable"
        class_name = str(item.get("class_name") or item.get("model_class_name") or "object")
        confidence = item.get("confidence")
        if confidence is not None:
            try:
                confidence_value = float(confidence)
            except (TypeError, ValueError, OverflowError):
                confidence_text = ""
            else:
                is_finite = bool(np.isfinite(confidence_value)) if np is not None else True
                confidence_text = f" {confidence_value:.0%}" if is_finite else ""
        else:
            confidence_text = ""
        position = str(item.get("horizontal_position") or "UNKNOWN").upper()
        label = f"{class_name}{confidence_text} | {track_label} | {position} | {category}"
        font_scale = max(0.35, min(0.65, preview_width / 1200))
        text_size, baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 2)
        label_baseline = max(text_size[1] + baseline + 4, y1)
        label_top = label_baseline - text_size[1] - baseline - 4
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
        cv2.rectangle(
            annotated,
            (x1, label_top),
            (min(preview_width - 1, x1 + text_size[0] + 8), label_baseline),
            color,
            cv2.FILLED,
        )
        cv2.putText(
            annotated,
            label,
            (x1 + 4, label_baseline - baseline - 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

    success, encoded = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 84])
    if not success:
        raise ValueError("annotated frame could not be encoded")
    return base64.b64encode(encoded.tobytes()).decode("ascii"), (preview_width, preview_height)
