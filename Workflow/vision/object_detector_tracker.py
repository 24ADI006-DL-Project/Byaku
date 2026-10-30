from collections.abc import Mapping
from numbers import Integral
from pathlib import Path
import json
import os
import shutil
from datetime import UTC, datetime
from typing import Any, Callable

# pyrefly: ignore [missing-import]
import numpy as np


DEFAULT_WEIGHTS = (
    Path(__file__).resolve().parents[2] / "Models" / "yolo11n.pt"
)


def _configure_ultralytics_settings() -> None:
    project_root = Path(__file__).resolve().parents[2]
    config_root = project_root / "Config" / "ultralytics"
    settings_path = config_root / "Ultralytics" / "settings.json"
    config_root.mkdir(parents=True, exist_ok=True)

    if settings_path.exists():
        try:
            settings_data = json.loads(settings_path.read_text(encoding="utf-8"))
            if not isinstance(settings_data, dict):
                raise ValueError("Ultralytics settings must be a JSON object")
        except (OSError, UnicodeError, json.JSONDecodeError, ValueError):
            timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
            backup_path = settings_path.with_name(f"settings.json.corrupt-{timestamp}.bak")
            shutil.copy2(settings_path, backup_path)

    os.environ["YOLO_CONFIG_DIR"] = str(config_root)


def _as_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    return np.asarray(value)


def _normalise_frame_id(value: Any) -> str:
    if isinstance(value, bool):
        raise ValueError("frame_id must be a non-empty string or non-negative integer")
    if isinstance(value, Integral) and int(value) >= 0:
        return str(int(value))
    if isinstance(value, str) and value.strip():
        return value.strip()
    raise ValueError("frame_id must be a non-empty string or non-negative integer")


def _normalise_frame_number(value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or int(value) < 1:
        raise ValueError("frame_number must be a positive integer (1-based)")
    return int(value)


class DetectionTrackingAdapter:
    """Adapt one persistent Ultralytics YOLO/ByteTrack model to BYAKU frame records."""

    def __init__(
        self,
        weights_path: str | Path = DEFAULT_WEIGHTS,
        *,
        confidence_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        imgsz: int = 640,
        tracker_config: str = "bytetrack.yaml",
        model: Any = None,
        model_factory: Callable[[str], Any] | None = None,
    ) -> None:
        if not np.isfinite(confidence_threshold) or not 0 <= confidence_threshold <= 1:
            raise ValueError("confidence_threshold must be between 0 and 1")
        if not np.isfinite(iou_threshold) or not 0 <= iou_threshold <= 1:
            raise ValueError("iou_threshold must be between 0 and 1")
        if not isinstance(tracker_config, str) or not tracker_config.strip():
            raise ValueError("tracker_config must be a non-empty string")

        if not isinstance(imgsz, int) or imgsz < 32:
            raise ValueError("imgsz must be an integer >= 32")
        self.weights_path = Path(weights_path).expanduser().resolve()
        self.confidence_threshold = float(confidence_threshold)
        self.iou_threshold = float(iou_threshold)
        self.imgsz = imgsz
        self.tracker_config = tracker_config
        self._last_frame_number: int | None = None
        self._seen_frame_ids: set[str] = set()

        if model is not None:
            self.model = model
            return
        if not self.weights_path.is_file():
            raise FileNotFoundError(f"YOLO weights not found: {self.weights_path}")

        if model_factory is None:
            try:
                _configure_ultralytics_settings()
                # pyrefly: ignore [missing-import]
                import ultralytics
                settings = getattr(ultralytics, "settings", None)
                YOLO = getattr(ultralytics, "YOLO", None)
                if settings is None or YOLO is None:
                    from ultralytics.utils import SETTINGS as settings  # type: ignore[no-redef]
                    from ultralytics.models.yolo import YOLO  # type: ignore[no-redef]
                if settings is None or YOLO is None or not callable(YOLO):
                    raise RuntimeError("Ultralytics YOLO is unavailable")
            except Exception as error:
                raise RuntimeError(
                    f"Ultralytics YOLO is unavailable: {type(error).__name__}: {error}"
                ) from error
            project_root = Path(__file__).resolve().parents[2]
            try:
                settings.update({
                    "datasets_dir": str(project_root / "Data" / "datasets"),
                    "weights_dir": str(project_root / "Models"),
                    "runs_dir": str(project_root / "Models" / "training_runs"),
                })
            except Exception as error:
                raise RuntimeError(
                    f"Could not initialize project Ultralytics settings: {type(error).__name__}: {error}"
                ) from error
            model_factory = YOLO

        try:
            if not callable(model_factory):
                raise RuntimeError("YOLO model factory is not callable")
            self.model = model_factory(str(self.weights_path))
        except Exception as error:
            raise RuntimeError(
                f"Could not load YOLO weights {self.weights_path}: {type(error).__name__}: {error}"
            ) from error

    @staticmethod
    def _error_result(
        frame_id: Any,
        frame_number: Any,
        status: str,
        message: str,
        model_path: str,
    ) -> dict[str, Any]:
        return {
            "frame": {"frame_id": frame_id, "frame_number": frame_number},
            "model": model_path,
            "status": status,
            "detections": [],
            "tracks": [],
            "orchestrator_detections": [],
            "empty_detections": True,
            "warnings": [message],
        }

    def _class_name(self, result: Any, class_id: int) -> str | None:
        names = getattr(result, "names", None) or getattr(self.model, "names", None)
        if isinstance(names, Mapping):
            name = names.get(class_id)
        elif isinstance(names, (list, tuple)) and 0 <= class_id < len(names):
            name = names[class_id]
        else:
            name = None
        if isinstance(name, str) and name.strip():
            return name.strip()
        return None

    def process_frame(
        self,
        frame: Any,
        *,
        frame_id: str | int,
        frame_number: int,
    ) -> dict[str, Any]:
        """Run YOLO detection plus persistent built-in ByteTrack on one image frame."""
        try:
            normalized_frame_id = _normalise_frame_id(frame_id)
            normalized_frame_number = _normalise_frame_number(frame_number)
            if normalized_frame_id in self._seen_frame_ids:
                raise ValueError("frame_id must be unique for each processed frame")
            if self._last_frame_number is not None and normalized_frame_number <= self._last_frame_number:
                raise ValueError("frame_number must increase for each processed frame")
            if not isinstance(frame, np.ndarray) or frame.ndim != 3 or frame.shape[2] < 3 or frame.size == 0:
                raise ValueError("frame must be a non-empty HxWxC NumPy image")
        except (TypeError, ValueError) as error:
            return self._error_result(frame_id, frame_number, "invalid_input", str(error), str(self.weights_path))

        height, width = frame.shape[:2]
        try:
            results = self.model.track(
                frame,
                conf=self.confidence_threshold,
                iou=self.iou_threshold,
                imgsz=self.imgsz,
                persist=True,
                tracker=self.tracker_config,
                verbose=False,
            )
        except Exception as error:
            return self._error_result(
                normalized_frame_id,
                normalized_frame_number,
                "error",
                f"YOLO/ByteTrack inference failed: {type(error).__name__}: {error}",
                str(self.weights_path),
            )

        if not isinstance(results, (list, tuple)) or not results:
            return self._error_result(
                normalized_frame_id,
                normalized_frame_number,
                "error",
                "YOLO/ByteTrack returned no result for the frame",
                str(self.weights_path),
            )

        result = results[0]
        boxes = getattr(result, "boxes", None)
        if boxes is None:
            return self._error_result(
                normalized_frame_id,
                normalized_frame_number,
                "error",
                "YOLO result did not contain boxes",
                str(self.weights_path),
            )

        try:
            coordinates = _as_numpy(boxes.xyxy).reshape((-1, 4))
            confidences = _as_numpy(boxes.conf).reshape((-1,))
            class_ids = _as_numpy(boxes.cls).reshape((-1,))
            raw_track_ids = getattr(boxes, "id", None)
            track_ids = None if raw_track_ids is None else _as_numpy(raw_track_ids).reshape((-1,))
        except Exception as error:
            return self._error_result(
                normalized_frame_id,
                normalized_frame_number,
                "error",
                f"Could not read YOLO/ByteTrack boxes: {type(error).__name__}: {error}",
                str(self.weights_path),
            )

        if not (len(coordinates) == len(confidences) == len(class_ids)):
            return self._error_result(
                normalized_frame_id,
                normalized_frame_number,
                "error",
                "YOLO box, confidence, and class arrays have different lengths",
                str(self.weights_path),
            )
        if track_ids is not None and len(track_ids) != len(coordinates):
            return self._error_result(
                normalized_frame_id,
                normalized_frame_number,
                "error",
                "ByteTrack ID array length does not match YOLO boxes",
                str(self.weights_path),
            )

        warnings: list[str] = []
        detections: list[dict[str, Any]] = []
        tracks: list[dict[str, Any]] = []
        orchestrator_detections: list[dict[str, Any]] = []
        untracked_count = 0

        for index, (raw_box, raw_confidence, raw_class_id) in enumerate(
            zip(coordinates, confidences, class_ids)
        ):
            try:
                box = [float(value) for value in raw_box]
                confidence = float(raw_confidence)
                class_id_float = float(raw_class_id)
                if not np.isfinite(class_id_float) or not class_id_float.is_integer():
                    raise ValueError("class ID is not a finite integer")
                class_id = int(class_id_float)
            except (TypeError, ValueError, OverflowError) as error:
                warnings.append(f"box {index} has invalid numeric data and was skipped: {error}")
                continue

            x1, y1, x2, y2 = box
            if (
                len(box) != 4
                or not all(np.isfinite(box))
                or not 0 <= confidence <= 1
                or x1 < 0
                or y1 < 0
                or x2 <= x1
                or y2 <= y1
                or x2 > width
                or y2 > height
            ):
                warnings.append(f"box {index} has an invalid confidence or xyxy box and was skipped")
                continue

            class_name = self._class_name(result, class_id)
            if class_name is None:
                warnings.append(f"box {index} has no class name for class ID {class_id} and was skipped")
                continue

            detection = {
                "class_name": class_name,
                "confidence": confidence,
                "bounding_box": box,
            }
            detections.append(detection)
            orchestrator_detection = dict(detection)

            track_id = None
            if track_ids is not None:
                try:
                    raw_id = float(track_ids[index])
                    if np.isfinite(raw_id) and raw_id.is_integer() and raw_id >= 0:
                        track_id = int(raw_id)
                    else:
                        warnings.append(f"box {index} has an invalid ByteTrack ID; left untracked")
                except (TypeError, ValueError, OverflowError):
                    warnings.append(f"box {index} has an invalid ByteTrack ID; left untracked")

            if track_id is None:
                untracked_count += 1
            else:
                tracked_record = {"track_id": track_id, **detection}
                tracks.append(tracked_record)
                orchestrator_detection["track_id"] = track_id
            orchestrator_detections.append(orchestrator_detection)

        if untracked_count:
            warnings.append(f"{untracked_count} detections had no usable track ID; identities were not invented")

        frame_metadata = {
            "frame_id": normalized_frame_id,
            "frame_number": normalized_frame_number,
            "image_shape": (height, width),
            "width": width,
            "height": height,
        }
        self._seen_frame_ids.add(normalized_frame_id)
        self._last_frame_number = normalized_frame_number
        return {
            "frame": frame_metadata,
            "model": str(self.weights_path),
            "thresholds": {
                "confidence": self.confidence_threshold,
                "iou": self.iou_threshold,
            },
            "tracker": self.tracker_config,
            "detections": detections,
            "tracks": tracks,
            "orchestrator_detections": orchestrator_detections,
            "empty_detections": not detections,
            "untracked_detection_count": untracked_count,
            "warnings": warnings,
            "status": "partial" if warnings else "success",
        }