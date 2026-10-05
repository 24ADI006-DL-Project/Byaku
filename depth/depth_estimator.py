from __future__ import annotations

from typing import Any

import cv2
import numpy as np
import torch
import torch.nn.functional as functional
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForDepthEstimation


class DepthEstimator:
    """Estimate normalized relative depth with a pretrained Depth Anything model."""

    def __init__(self, model_name: str = "depth-anything/Depth-Anything-V2-Small-hf") -> None:
        self.model_name = model_name
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.processor = AutoImageProcessor.from_pretrained(model_name)
        self.model = AutoModelForDepthEstimation.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()

    def estimate_depth(self, frame: Any) -> np.ndarray | None:
        """Return a normalized relative-depth map matching the input frame size."""
        if not isinstance(frame, np.ndarray) or frame.ndim != 3 or frame.size == 0:
            return None

        height, width = frame.shape[:2]
        if height <= 0 or width <= 0:
            return None

        try:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            inputs = self.processor(images=Image.fromarray(rgb), return_tensors="pt")
            inputs = {name: value.to(self.device) for name, value in inputs.items()}
            with torch.inference_mode():
                prediction = self.model(**inputs).predicted_depth
                prediction = functional.interpolate(
                    prediction.unsqueeze(1),
                    size=(height, width),
                    mode="bicubic",
                    align_corners=False,
                ).squeeze(0).squeeze(0)
            return self.normalize_depth_map(prediction.detach().cpu().numpy())
        except Exception:
            return None

    @staticmethod
    def normalize_depth_map(depth_map: Any) -> np.ndarray:
        """Normalize finite depth predictions to the model's relative 0-to-1 scale."""
        values = np.asarray(depth_map, dtype=np.float32)
        if values.ndim != 2:
            return np.empty((0, 0), dtype=np.float32)

        finite = np.isfinite(values)
        if not finite.any():
            return np.zeros(values.shape, dtype=np.float32)

        minimum = float(values[finite].min())
        maximum = float(values[finite].max())
        normalized = np.zeros(values.shape, dtype=np.float32)
        if maximum > minimum:
            normalized[finite] = (values[finite] - minimum) / (maximum - minimum)
        return normalized

    @staticmethod
    def get_depth_at_pixel(depth_map: Any, x: int, y: int) -> float | None:
        """Return one relative-depth value, or None when coordinates are invalid."""
        if not isinstance(depth_map, np.ndarray) or depth_map.ndim != 2:
            return None
        height, width = depth_map.shape
        if x < 0 or y < 0 or x >= width or y >= height:
            return None
        value = float(depth_map[y, x])
        return value if np.isfinite(value) else None