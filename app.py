from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import cv2
import numpy as np


@dataclass(frozen=True)
class AppConfig:
    yolo_model: str = "yolo11n.pt"
    depth_model: str = "depth-anything/Depth-Anything-V2-Small-hf"
    confidence_threshold: float = 0.5
    processing_fps: float = 5.0
    frame_skip: int = 0

    def __post_init__(self) -> None:
        if self.processing_fps <= 0:
            raise ValueError("processing_fps must be greater than zero")
        if self.frame_skip < 0:
            raise ValueError("frame_skip must be non-negative")
        if not 0 <= self.confidence_threshold <= 1:
            raise ValueError("confidence_threshold must be between zero and one")

    @classmethod
    def from_environment(cls) -> AppConfig:
        return cls(
            yolo_model=os.getenv("YOLO_MODEL", "yolo11n.pt"),
            depth_model=os.getenv(
                "DEPTH_MODEL", "depth-anything/Depth-Anything-V2-Small-hf"
            ),
            confidence_threshold=float(os.getenv("DETECTION_CONFIDENCE", "0.5")),
            processing_fps=float(os.getenv("PROCESSING_FPS", "5")),
            frame_skip=int(os.getenv("FRAME_SKIP", "0")),
        )


class VisionAssistantApp:
    """Connect the existing vision, scene, query, narration, and speech modules."""

    def __init__(
        self,
        config: AppConfig | None = None,
        *,
        detector: Any = None,
        tracker: Any = None,
        depth_estimator: Any = None,
        depth_integrator: Any = None,
        spatial_processor: Any = None,
        scene_memory: Any = None,
        query_processor: Any = None,
        narrator: Any = None,
        tts: Any = None,
        speech_input: Any = None,
        event_detector: Any = None,
        camera_factory: Callable[..., Any] | None = None,
        speak_events: bool = True,
    ) -> None:
        from narration.narrator import Narrator
        from query.query_processor import QueryProcessor
        from query.spatial_query import SpatialQueryEngine
        from reasoning.events import EventDetector
        from scene_graph.memory import SceneMemory
        from spatial.depth_integration import DepthIntegrator
        from spatial.spatial_processor import SpatialProcessor
        from speech.speech_input import SpeechInput
        from speech.tts import TextToSpeech
        from tracking.tracker import ObjectTracker

        self.config = config or AppConfig.from_environment()
        self.component_errors: dict[str, str] = {}
        self.detector = detector or self._create_detector()
        self.tracker = tracker or self._create_component(
            "tracker", ObjectTracker, track_buffer=30
        )
        self.depth_estimator = depth_estimator or self._create_depth_estimator()
        self.depth_integrator = depth_integrator or DepthIntegrator()
        self.spatial_processor = spatial_processor or SpatialProcessor()
        self.scene_memory = scene_memory or SceneMemory()
        self.query_processor = query_processor or QueryProcessor()
        self.query_engine = SpatialQueryEngine(self.scene_memory, self.query_processor)
        self.narrator = narrator or Narrator()
        self.tts = tts or TextToSpeech()
        self.speech_input = speech_input or SpeechInput()
        self.event_detector = event_detector or EventDetector()
        self.speak_events = speak_events
        self.camera_factory = camera_factory
        if self.camera_factory is None:
            from camera.camera import CameraInput

            self.camera_factory = CameraInput
        self._state_lock = threading.RLock()
        self._tts_lock = threading.Lock()
        self._speech_thread: threading.Thread | None = None
        self._closed = False

    def _create_component(self, name: str, factory: Callable[..., Any], **kwargs: Any) -> Any:
        try:
            return factory(**kwargs)
        except Exception as exc:
            self.component_errors[name] = str(exc) or type(exc).__name__
            return None

    def _create_detector(self) -> Any:
        try:
            from detection.detector import YOLODetector

            return self._create_component(
                "detector",
                YOLODetector,
                model_name=self.config.yolo_model,
                confidence_threshold=self.config.confidence_threshold,
            )
        except Exception as exc:
            self.component_errors["detector"] = str(exc) or type(exc).__name__
            return None

    def _create_depth_estimator(self) -> Any:
        try:
            from depth.depth_estimator import DepthEstimator

            return self._create_component(
                "depth_estimator", DepthEstimator, model_name=self.config.depth_model
            )
        except Exception as exc:
            self.component_errors["depth_estimator"] = str(exc) or type(exc).__name__
            return None

    @staticmethod
    def _empty_result(message: str) -> dict[str, Any]:
        return {
            "frame": None,
            "video": None,
            "detections": [],
            "tracks": [],
            "depth_map": None,
            "objects": [],
            "scene": {"objects": [], "relationships": []},
            "events": [],
            "status": message,
        }

    def process_frame(self, frame: Any) -> dict[str, Any]:
        """Run one OpenCV frame through the already-initialized modules."""
        if not isinstance(frame, np.ndarray) or frame.size == 0:
            return self._empty_result("No frame available")

        with self._state_lock:
            if self._closed:
                return self._empty_result("Application is shut down")
            errors: list[str] = []
            try:
                detections = self.detector.detect(frame) if self.detector else []
            except Exception as exc:
                detections = []
                errors.append(f"Detection failed: {exc}")

            try:
                tracks = self.tracker.update(detections) if self.tracker else []
            except Exception as exc:
                tracks = []
                errors.append(f"Tracking failed: {exc}")

            depth_map = None
            if self.depth_estimator is not None:
                try:
                    depth_map = self.depth_estimator.estimate_depth(frame)
                except Exception as exc:
                    errors.append(f"Depth estimation failed: {exc}")

            try:
                enriched = self.depth_integrator.integrate(
                    tracks, depth_map, image_shape=frame.shape
                )
                objects = self.spatial_processor.process(enriched, frame.shape)
                current_scene = self.scene_memory.update(objects)
                events = self.event_detector.update(objects)
            except Exception as exc:
                objects = []
                current_scene = self.scene_memory.objects
                events = []
                errors.append(f"Spatial processing failed: {exc}")

            event_objects = {obj.get("track_id"): obj for obj in objects}
            for event in events:
                event_object = event_objects.get(event.get("track_id"), {})
                event["narration"] = self._narrate_event(event, event_object)
                if event.get("message_required") and self.speak_events:
                    self._schedule_speech(event["narration"])

            output_frame = self._draw_objects(frame, objects)
            scene_graph = getattr(self.scene_memory, "scene_graph", None)
            relationships = getattr(scene_graph, "relationships", [])
            status = "; ".join(errors) if errors else "Ready"
            if self.detector is None:
                status = self.component_errors.get("detector", "Object detector unavailable")
            return {
                "frame": output_frame,
                "detections": detections,
                "tracks": tracks,
                "depth_map": depth_map,
                "objects": current_scene,
                "scene": {"objects": current_scene, "relationships": relationships},
                "events": events,
                "status": status,
            }

    def process_video(self, video_path: str | os.PathLike[str]) -> dict[str, Any]:
        """Process a video file at the configured rate and write an annotated copy."""
        if not video_path or not Path(video_path).is_file():
            return self._empty_result("Video file could not be opened")
        if self._closed:
            return self._empty_result("Application is shut down")

        camera = None
        writer = None
        output_path: str | None = None
        latest = self._empty_result("No frames were read from the video")
        try:
            camera = self.camera_factory(video_path, frame_skip=self.config.frame_skip)
            if not camera.is_open:
                return self._empty_result("Video file could not be opened")
            source_fps = float(camera.cap.get(cv2.CAP_PROP_FPS) or 0.0)
            if source_fps > 0:
                camera.frame_skip = max(
                    self.config.frame_skip,
                    int(source_fps / self.config.processing_fps) - 1,
                )
            output_fps = min(source_fps, self.config.processing_fps) if source_fps > 0 else self.config.processing_fps

            while True:
                item = camera.read()
                if item is None:
                    break
                latest = self.process_frame(item["frame"])
                if latest["frame"] is None:
                    continue
                if writer is None:
                    height, width = latest["frame"].shape[:2]
                    handle, output_path = tempfile.mkstemp(
                        prefix="vision_assistant_", suffix=".mp4"
                    )
                    os.close(handle)
                    writer = cv2.VideoWriter(
                        output_path,
                        cv2.VideoWriter_fourcc(*"mp4v"),
                        output_fps,
                        (width, height),
                    )
                    if not writer.isOpened():
                        writer.release()
                        writer = None
                        os.unlink(output_path)
                        output_path = None
                if writer is not None:
                    writer.write(latest["frame"])
            if output_path and writer is not None:
                latest["video"] = output_path
            else:
                latest["video"] = None
            return latest
        except Exception as exc:
            latest["status"] = f"Video processing failed: {exc}"
            latest["video"] = None
            return latest
        finally:
            if writer is not None:
                writer.release()
            if camera is not None:
                camera.release()

    def answer_query(self, query: str) -> dict[str, Any]:
        """Answer against current scene memory, narrate it, and prepare audio."""
        with self._state_lock:
            if self._closed:
                result = {"intent": None, "found": False, "target": None, "objects": [], "status": "unavailable"}
                narration = "The application is shut down."
            else:
                result = self.query_engine.answer(query)
                narration = self.narrator.narrate(result)
        return {
            "query_result": result,
            "narration": narration,
            "audio": self._synthesize_audio(narration),
        }

    def listen_for_query(self) -> dict[str, Any]:
        """Use the configured speech-input module and answer its transcript."""
        transcript = self.speech_input.listen() if self.speech_input else None
        if not transcript:
            reason = getattr(self.speech_input, "last_status", "unavailable")
            message = getattr(self.speech_input, "last_error", None)
            text = f"Voice input unavailable ({reason})"
            if message:
                text = f"{text}: {message}"
            return {"transcript": "", "query_result": None, "narration": text, "audio": None}
        answer = self.answer_query(transcript)
        answer["transcript"] = transcript
        return answer

    def _synthesize_audio(self, text: str) -> str | None:
        engine = getattr(self.tts, "engine", None)
        save_to_file = getattr(engine, "save_to_file", None)
        run_and_wait = getattr(engine, "runAndWait", None)
        if not callable(save_to_file) or not callable(run_and_wait):
            return None
        if self._speech_thread is not None and self._speech_thread.is_alive():
            return None

        handle, audio_path = tempfile.mkstemp(prefix="vision_narration_", suffix=".wav")
        os.close(handle)
        try:
            with self._tts_lock:
                save_to_file(text, audio_path)
                run_and_wait()
            if not os.path.isfile(audio_path) or os.path.getsize(audio_path) == 0:
                os.unlink(audio_path)
                return None
            return audio_path
        except Exception:
            if os.path.exists(audio_path):
                os.unlink(audio_path)
            return None

    def _schedule_speech(self, text: str) -> None:
        if not text or self.tts is None:
            return
        with self._tts_lock:
            if self._speech_thread is not None and self._speech_thread.is_alive():
                return

            def speak() -> None:
                try:
                    self.tts.speak(text)
                except Exception:
                    pass

            self._speech_thread = threading.Thread(target=speak, daemon=True)
            self._speech_thread.start()

    def _narrate_event(self, event: dict[str, Any], spatial_object: dict[str, Any]) -> str:
        target = str(event.get("class_name", "object"))
        if event.get("event_type") == "object_disappeared":
            return self.narrator.narrate(
                {"intent": "existence", "found": False, "target": target, "objects": []}
            )
        return self.narrator.narrate(
            {
                "intent": "locate",
                "found": bool(spatial_object),
                "target": target,
                "objects": [spatial_object] if spatial_object else [],
            }
        )

    @staticmethod
    def _draw_objects(frame: np.ndarray, objects: list[dict[str, Any]]) -> np.ndarray:
        output = frame.copy()
        for obj in objects:
            box = obj.get("bounding_box")
            if not isinstance(box, (list, tuple)) or len(box) != 4:
                continue
            x1, y1, x2, y2 = (int(value) for value in box)
            label = str(obj.get("class_name", "object"))
            position = obj.get("horizontal_position")
            distance = obj.get("distance_category")
            details = ", ".join(str(value) for value in (position, distance) if value)
            if details:
                label = f"{label}: {details}"
            cv2.rectangle(output, (x1, y1), (x2, y2), (40, 190, 90), 2)
            cv2.putText(
                output,
                label,
                (x1, max(18, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (40, 190, 90),
                2,
                cv2.LINE_AA,
            )
        return output

    def close(self) -> None:
        """Release per-application state and prevent subsequent processing."""
        with self._state_lock:
            if self._closed:
                return
            self._closed = True
            reset_tracker = getattr(self.tracker, "reset", None)
            if callable(reset_tracker):
                reset_tracker()
            reset_memory = getattr(self.scene_memory, "reset", None)
            if callable(reset_memory):
                reset_memory()
            reset_events = getattr(self.event_detector, "reset", None)
            if callable(reset_events):
                reset_events()


def main() -> None:
    server_path = Path(__file__).parent / "web" / "server" / "server.js"
    subprocess.run(["node", str(server_path)], check=True)


if __name__ == "__main__":
    main()
