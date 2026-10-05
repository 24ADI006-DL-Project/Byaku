import os
import tempfile

import cv2
import numpy as np

from camera.camera import CameraInput


def create_test_video(path, frames=5, width=64, height=64):
    writer = cv2.VideoWriter(
        path,
        cv2.VideoWriter_fourcc(*"mp4v"),
        10,
        (width, height),
    )
    assert writer.isOpened(), "Failed to open temporary video writer"

    for i in range(frames):
        frame = 50 * (i + 1)
        color = (frame % 255, (frame * 3) % 255, (frame * 5) % 255)
        image = np.full((height, width, 3), 0, dtype=np.uint8)
        image[:] = color
        writer.write(image)

    writer.release()
    return path


def test_camera_reads_frames_and_metadata():
    with tempfile.TemporaryDirectory() as tmpdir:
        video_path = os.path.join(tmpdir, "test_camera.mp4")
        create_test_video(video_path, frames=5)

        camera = CameraInput(video_path, fps=10, frame_skip=0, resize=(32, 32))
        first = camera.read()
        second = camera.read()

        assert first["frame"] is not None
        assert first["frame"].shape[:2] == (32, 32)
        assert isinstance(first["timestamp"], float)
        assert first["frame_id"] == 0
        assert second["frame_id"] == 1

        camera.release()


def test_camera_handles_missing_video_gracefully():
    camera = CameraInput("does_not_exist.mp4")
    result = camera.read()
    assert result is None
    camera.release()
