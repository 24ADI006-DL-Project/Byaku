import time
from typing import Optional, Tuple, Union

import cv2


class CameraInput:
    """Thin wrapper around OpenCV video capture for webcam or file input.

    The class is intentionally independent from detection and depth logic. It only
    provides frames, timestamps, and frame IDs while supporting optional frame
    skipping and resize operations.
    """

    def __init__(
        self,
        source: Union[int, str] = 0,
        fps: Optional[float] = None,
        frame_skip: int = 0,
        resize: Optional[Tuple[int, int]] = None,
    ) -> None:
        self.source = source
        self.target_fps = fps
        self.frame_skip = max(0, int(frame_skip))
        self.resize = resize
        self.frame_id = 0
        self.cap = cv2.VideoCapture(source)

        if not self.cap.isOpened():
            self.cap = None
            return

        if fps is not None:
            self.cap.set(cv2.CAP_PROP_FPS, float(fps))

    def read(self):
        """Return the next processed frame as a dictionary with metadata.

        Returns:
            dict | None: {"frame": ..., "timestamp": float, "frame_id": int}
                or None if no frame is available.
        """
        if self.cap is None or not self.cap.isOpened():
            return None

        for _ in range(self.frame_skip):
            if not self.cap.grab():
                return None

        ok, frame = self.cap.read()
        if not ok or frame is None:
            return None

        if self.resize is not None:
            frame = cv2.resize(frame, self.resize, interpolation=cv2.INTER_AREA)

        result = {
            "frame": frame,
            "timestamp": time.time(),
            "frame_id": self.frame_id,
        }
        self.frame_id += 1
        return result

    def release(self) -> None:
        if self.cap is not None and self.cap.isOpened():
            self.cap.release()
        self.cap = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.release()

    @property
    def is_open(self) -> bool:
        return self.cap is not None and self.cap.isOpened()
