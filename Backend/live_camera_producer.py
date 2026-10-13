import base64
from collections.abc import Callable, Mapping
import time
from typing import Any
from uuid import uuid4

import cv2
import numpy as np

from .fastapi_application import FrameRequest, app, process_live_frame


def _latest_committed_frame_number() -> int:
    orchestrator = getattr(app.state, "orchestrator", None)
    value = getattr(orchestrator, "_latest_frame_number", None)
    return value if isinstance(value, int) and not isinstance(value, bool) and value > 0 else 0


class LiveCameraProducer:
    """Synchronously capture camera frames and submit them through the live frame API."""

    def __init__(
        self,
        source: int | str = 0,
        *,
        capture_factory: Callable[[int | str], Any] | None = None,
        frame_submitter: Callable[[FrameRequest], Mapping[str, Any]] | None = None,
        frame_number_provider: Callable[[], int] | None = None,
        max_consecutive_read_failures: int = 3,
        jpeg_quality: int = 90,
    ) -> None:
        if (
            isinstance(max_consecutive_read_failures, bool)
            or not isinstance(max_consecutive_read_failures, int)
            or max_consecutive_read_failures < 1
        ):
            raise ValueError("max_consecutive_read_failures must be a positive integer")
        if isinstance(jpeg_quality, bool) or not isinstance(jpeg_quality, int) or not 1 <= jpeg_quality <= 100:
            raise ValueError("jpeg_quality must be an integer from 1 to 100")

        self.source = source
        self.capture_factory = capture_factory or cv2.VideoCapture
        self.frame_submitter = frame_submitter or process_live_frame
        self.frame_number_provider = frame_number_provider or _latest_committed_frame_number
        self.max_consecutive_read_failures = max_consecutive_read_failures
        self.jpeg_quality = jpeg_quality
        self._producer_id = uuid4().hex
        self._frame_number = 0
        self._consecutive_read_failures = 0
        self._capture: Any = None
        self._running = False

    @property
    def is_running(self) -> bool:
        return self._running

    @property
    def frame_number(self) -> int:
        return self._frame_number

    def _release_capture(self) -> str | None:
        capture = self._capture
        self._capture = None
        self._running = False
        if capture is None:
            return None
        try:
            capture.release()
        except Exception as error:
            return f"camera release failed: {type(error).__name__}: {error}"
        return None

    def _stop_after_error(self, *, status: str, message: str) -> dict[str, Any]:
        release_warning = self._release_capture()
        result: dict[str, Any] = {"status": status, "message": message, "recoverable": False}
        if release_warning:
            result["warnings"] = [release_warning]
        return result

    def start(self) -> dict[str, Any]:
        if self._running:
            return {"status": "already_running", "source": self.source}

        try:
            capture = self.capture_factory(self.source)
            self._capture = capture
            if capture is None or not capture.isOpened():
                release_warning = self._release_capture()
                result: dict[str, Any] = {
                    "status": "unavailable",
                    "message": f"camera source {self.source!r} could not be opened",
                }
                if release_warning:
                    result["warnings"] = [release_warning]
                return result
        except Exception as error:
            return self._stop_after_error(
                status="camera_error",
                message=f"camera open failed: {type(error).__name__}: {error}",
            )

        try:
            latest_frame_number = self.frame_number_provider()
        except Exception as error:
            return self._stop_after_error(
                status="camera_error",
                message=f"could not read latest frame number: {type(error).__name__}: {error}",
            )
        if isinstance(latest_frame_number, int) and not isinstance(latest_frame_number, bool):
            self._frame_number = max(self._frame_number, latest_frame_number)

        self._running = True
        self._consecutive_read_failures = 0
        return {"status": "started", "source": self.source, "next_frame_number": self._frame_number + 1}

    def stop(self) -> dict[str, Any]:
        was_running = self._running
        release_warning = self._release_capture()
        result: dict[str, Any] = {"status": "stopped", "already_stopped": not was_running}
        if release_warning:
            result["warnings"] = [release_warning]
        return result

    def capture_once(self) -> dict[str, Any]:
        if not self._running or self._capture is None:
            return {"status": "not_running", "message": "camera producer is not running"}

        try:
            read_ok, frame = self._capture.read()
        except Exception as error:
            return self._stop_after_error(
                status="capture_error",
                message=f"camera read failed: {type(error).__name__}: {error}",
            )

        if (
            not read_ok
            or not isinstance(frame, np.ndarray)
            or frame.ndim != 3
            or frame.shape[2] != 3
            or frame.size == 0
            or frame.dtype != np.uint8
        ):
            self._consecutive_read_failures += 1
            stopped = self._consecutive_read_failures >= self.max_consecutive_read_failures
            if stopped:
                cleanup = self.stop()
            else:
                cleanup = None
            result: dict[str, Any] = {
                "status": "read_failed",
                "message": "camera did not provide a valid uint8 HxWx3 frame",
                "consecutive_failures": self._consecutive_read_failures,
                "recoverable": not stopped,
            }
            if cleanup and cleanup.get("warnings"):
                result["warnings"] = cleanup["warnings"]
            return result

        self._consecutive_read_failures = 0
        try:
            encoded_ok, encoded = cv2.imencode(
                ".jpg",
                frame,
                [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality],
            )
            if not encoded_ok:
                return self._stop_after_error(
                    status="encode_failed",
                    message="camera frame could not be encoded as JPEG",
                )

            height, width = frame.shape[:2]
            self._frame_number += 1
            request = FrameRequest(
                frame_id=f"{self._producer_id}:{self._frame_number}",
                frame_number=self._frame_number,
                width=width,
                height=height,
                data=base64.b64encode(encoded.tobytes()).decode("ascii"),
                timestamp=time.time(),
            )
            response = self.frame_submitter(request)
        except Exception as error:
            return self._stop_after_error(
                status="submission_failed",
                message=f"frame submission failed: {type(error).__name__}: {error}",
            )

        if (
            not isinstance(response, Mapping)
            or response.get("type") != "frame_result"
            or response.get("status") not in {"success", "partial"}
        ):
            return self._stop_after_error(
                status="submission_failed",
                message="frame API did not return a successful frame_result",
            )

        return {
            "status": "submitted",
            "frame_id": request.frame_id,
            "frame_number": request.frame_number,
            "frame_result": dict(response),
        }

    def run(self, *, max_frames: int | None = None) -> dict[str, Any]:
        if max_frames is not None and (
            isinstance(max_frames, bool) or not isinstance(max_frames, int) or max_frames < 1
        ):
            return {"status": "invalid_input", "message": "max_frames must be a positive integer or None"}

        start_result = self.start()
        if not self._running:
            return {"status": start_result["status"], "results": [start_result], "frames_submitted": 0}

        results: list[dict[str, Any]] = []
        frames_submitted = 0
        terminal_status = "completed"
        try:
            while self._running and (max_frames is None or frames_submitted < max_frames):
                result = self.capture_once()
                results.append(result)
                if result.get("status") == "submitted":
                    frames_submitted += 1
                elif result.get("status") == "read_failed" and result.get("recoverable"):
                    continue
                else:
                    terminal_status = result.get("status", "error")
                    break

            if max_frames is not None and frames_submitted >= max_frames:
                terminal_status = "completed"
        finally:
            if self._running:
                self.stop()

        return {
            "status": terminal_status,
            "results": results,
            "frames_submitted": frames_submitted,
        }