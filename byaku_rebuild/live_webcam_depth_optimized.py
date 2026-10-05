import argparse
from pathlib import Path
import statistics
import threading
import time


MODEL_NAME = "depth-anything/Depth-Anything-V2-Small-hf"
PREVIEW_DIR = Path(__file__).resolve().parent / "results"


class LatestFrameCapture:
    def __init__(self, camera: object) -> None:
        self.camera = camera
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._frame = None
        self._sequence = 0
        self._captured_at = 0.0
        self._read_failures = 0
        self._error: str | None = None
        self._thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()

    def _capture_loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                success, frame = self.camera.read()
            except Exception as error:
                with self._lock:
                    self._error = f"{type(error).__name__}: {error}"
                self._stop_event.set()
                return

            if not success or frame is None:
                with self._lock:
                    self._read_failures += 1
                self._stop_event.wait(0.02)
                continue

            with self._lock:
                self._sequence += 1
                self._frame = frame
                self._captured_at = time.perf_counter()

    def latest_after(self, sequence: int) -> tuple[int, object | None, float, int, str | None]:
        with self._lock:
            return self._sequence, self._frame, self._captured_at, self._read_failures, self._error

    def stop(self) -> None:
        self._stop_event.set()
        self.camera.release()
        self._thread.join(timeout=2.0)


def unique_preview_path() -> Path:
    path = PREVIEW_DIR / "live_webcam_depth_optimized_preview.png"
    if not path.exists():
        return path

    version = 2
    while True:
        candidate = PREVIEW_DIR / f"live_webcam_depth_optimized_preview_{version}.png"
        if not candidate.exists():
            return candidate
        version += 1


def main() -> int:
    parser = argparse.ArgumentParser(description="Optimized live Depth Anything V2 webcam test (CPU only).")
    parser.add_argument("--test-seconds", type=float, help="Stop after this many seconds; otherwise press Q")
    parser.add_argument(
        "--input-size",
        type=int,
        default=364,
        help="Square processor target size; must be divisible by 14 (default: 364, model default: 518)",
    )
    parser.add_argument(
        "--input-format",
        choices=("numpy", "pil"),
        default="numpy",
        help="Pass the RGB array directly to the processor or wrap it as a PIL image",
    )
    parser.add_argument(
        "--synchronous-capture",
        action="store_true",
        help="Read one camera frame in the inference loop instead of using a latest-frame thread",
    )
    parser.add_argument(
        "--fresh-buffers",
        action="store_true",
        help="Allocate visualization buffers each frame instead of reusing them",
    )
    parser.add_argument("--save-preview", action="store_true", help="Save one comparison frame for visual inspection")
    args = parser.parse_args()
    if args.test_seconds is not None and args.test_seconds <= 0:
        parser.error("--test-seconds must be greater than zero")
    if args.input_size < 112 or args.input_size % 14 != 0:
        parser.error("--input-size must be at least 112 and divisible by 14")

    try:
        import cv2
        import numpy as np
        import torch
        import torch.nn.functional as functional
        from PIL import Image
        from transformers import AutoImageProcessor, AutoModelForDepthEstimation
    except Exception as error:
        print(f"ERROR: Depth estimation dependency unavailable: {type(error).__name__}: {error}")
        return 1

    device = torch.device("cpu")
    try:
        processor = AutoImageProcessor.from_pretrained(MODEL_NAME, local_files_only=True)
        model = AutoModelForDepthEstimation.from_pretrained(MODEL_NAME, local_files_only=True)
        model.to(device)
        model.eval()
    except Exception as error:
        print(
            f"ERROR: Could not load cached model {MODEL_NAME!r} (network downloads disabled): "
            f"{type(error).__name__}: {error}"
        )
        return 1

    try:
        camera = cv2.VideoCapture(0)
    except Exception as error:
        print(f"ERROR: Could not initialize default webcam: {type(error).__name__}: {error}")
        return 1
    if not camera.isOpened():
        camera.release()
        print("ERROR: Could not open the default webcam. Check that it is connected and available.")
        return 1

    camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    capture_worker = None if args.synchronous_capture else LatestFrameCapture(camera)
    normalization_range: tuple[float, float] | None = None
    inference_times_ms: list[float] = []
    captured_frames = 0
    successful_frames = 0
    inference_failures = 0
    capture_failures = 0
    dropped_camera_frames = 0
    capture_ages_ms: list[float] = []
    first_inference_error: str | None = None
    preview_path: Path | None = None
    previous_sequence = 0
    last_new_frame_at = time.perf_counter()
    started_at = time.perf_counter()
    run_completed_at = started_at
    next_report_at = 5.0
    status = 0
    rgb_buffer = None
    normalized_buffer = None
    depth_u8_buffer = None
    depth_visualization_buffer = None
    comparison_buffer = None

    print(f"Model: {MODEL_NAME}; device: CPU; local cache only.")
    print(
        f"Settings: input={args.input_size}, format={args.input_format}, "
        f"capture={'threaded latest-frame' if capture_worker else 'synchronous'}, "
        f"display-buffers={'fresh' if args.fresh_buffers else 'reused'}, "
        "torch.inference_mode=True."
    )
    print("Live depth running. Press Q in the preview window to stop.")

    try:
        while True:
            if capture_worker is None:
                capture_ok, frame = camera.read()
                if not capture_ok or frame is None:
                    capture_failures += 1
                    print("ERROR: Webcam stopped returning readable frames.")
                    status = 1
                    break
                captured_frames += 1
                frame_sequence = captured_frames
                captured_at = time.perf_counter()
                read_failures = 0
                capture_error = None
            else:
                frame_sequence, frame, captured_at, read_failures, capture_error = capture_worker.latest_after(
                    previous_sequence
                )
                capture_failures = read_failures
                if frame_sequence <= previous_sequence or frame is None:
                    if capture_error is not None:
                        print(f"ERROR: Webcam capture thread failed: {capture_error}")
                        status = 1
                        break
                    if time.perf_counter() - last_new_frame_at > 3.0:
                        print("ERROR: Webcam capture thread produced no new frame for 3 seconds.")
                        status = 1
                        break
                    time.sleep(0.002)
                    if args.test_seconds is not None and time.perf_counter() - started_at >= args.test_seconds:
                        break
                    continue

                if previous_sequence:
                    dropped_camera_frames += max(0, frame_sequence - previous_sequence - 1)
                previous_sequence = frame_sequence
                captured_frames = frame_sequence
                last_new_frame_at = time.perf_counter()

            height, width = frame.shape[:2]
            frame_started_at = time.perf_counter()
            capture_ages_ms.append(max(0.0, frame_started_at - captured_at) * 1000)
            try:
                if args.fresh_buffers:
                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                else:
                    if rgb_buffer is None or rgb_buffer.shape != frame.shape:
                        rgb_buffer = np.empty_like(frame)
                    cv2.cvtColor(frame, cv2.COLOR_BGR2RGB, dst=rgb_buffer)
                    rgb_frame = rgb_buffer
                if args.input_format == "pil":
                    model_image = Image.fromarray(rgb_frame)
                else:
                    model_image = rgb_frame
                inputs = processor(
                    images=model_image,
                    return_tensors="pt",
                    size={"height": args.input_size, "width": args.input_size},
                )
                inputs = {name: tensor.to(device) for name, tensor in inputs.items()}
                with torch.inference_mode():
                    prediction = model(**inputs).predicted_depth
                    prediction = functional.interpolate(
                        prediction.unsqueeze(1),
                        size=(height, width),
                        mode="bicubic",
                        align_corners=False,
                    ).squeeze(0).squeeze(0)
                depth_values = prediction.detach().cpu().numpy()
                finite_values = depth_values[np.isfinite(depth_values)]
                if finite_values.size == 0:
                    raise ValueError("model produced no finite depth values")
                if normalization_range is None:
                    lower, upper = (float(value) for value in np.percentile(finite_values, (2, 98)))
                    if upper <= lower:
                        lower = float(finite_values.min())
                        upper = float(finite_values.max())
                    if upper <= lower:
                        raise ValueError("first depth frame has no usable value range")
                    normalization_range = (lower, upper)

                inference_ms = (time.perf_counter() - frame_started_at) * 1000
                inference_times_ms.append(inference_ms)
                if args.fresh_buffers:
                    normalized = np.clip(
                        (depth_values - normalization_range[0])
                        / (normalization_range[1] - normalization_range[0]),
                        0.0,
                        1.0,
                    )
                    depth_u8 = (normalized * 255).astype(np.uint8)
                    depth_visualization = cv2.applyColorMap(depth_u8, cv2.COLORMAP_INFERNO)
                else:
                    if normalized_buffer is None or normalized_buffer.shape != depth_values.shape:
                        normalized_buffer = np.empty(depth_values.shape, dtype=np.float32)
                        depth_u8_buffer = np.empty(depth_values.shape, dtype=np.uint8)
                        depth_visualization_buffer = np.empty((height, width, 3), dtype=np.uint8)
                    np.subtract(depth_values, normalization_range[0], out=normalized_buffer)
                    normalized_buffer *= 1.0 / (normalization_range[1] - normalization_range[0])
                    np.clip(normalized_buffer, 0.0, 1.0, out=normalized_buffer)
                    normalized_buffer *= 255.0
                    np.copyto(depth_u8_buffer, normalized_buffer, casting="unsafe")
                    depth_visualization = cv2.applyColorMap(
                        depth_u8_buffer,
                        cv2.COLORMAP_INFERNO,
                        depth_visualization_buffer,
                    )
                successful_frames += 1
                frame_error = None
            except Exception as error:
                inference_failures += 1
                frame_error = f"{type(error).__name__}: {error}"
                if first_inference_error is None:
                    first_inference_error = frame_error
                    print(f"ERROR: Depth inference failed on captured frame {captured_frames}: {frame_error}")
                depth_visualization = np.zeros_like(frame)
                cv2.putText(
                    depth_visualization,
                    "Depth inference failed",
                    (12, max(30, height // 2)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )

            elapsed = time.perf_counter() - started_at
            current_fps = successful_frames / elapsed if elapsed > 0 else 0.0
            displayed_inference_ms = inference_times_ms[-1] if inference_times_ms else 0.0
            if args.fresh_buffers:
                comparison = np.concatenate((frame, depth_visualization), axis=1)
            else:
                if comparison_buffer is None or comparison_buffer.shape != (height, width * 2, 3):
                    comparison_buffer = np.empty((height, width * 2, 3), dtype=np.uint8)
                comparison_buffer[:, :width] = frame
                comparison_buffer[:, width:] = depth_visualization
                comparison = comparison_buffer
            cv2.putText(
                comparison,
                f"Inference: {displayed_inference_ms:.0f} ms | FPS: {current_fps:.2f}",
                (10, 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                comparison,
                "Relative depth (fixed scale)",
                (width + 10, 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

            if args.save_preview and preview_path is None and frame_error is None:
                try:
                    PREVIEW_DIR.mkdir(parents=True, exist_ok=True)
                    preview_path = unique_preview_path()
                    if not cv2.imwrite(str(preview_path), comparison):
                        raise OSError(f"OpenCV could not write {preview_path}")
                    print(f"Single preview snapshot: {preview_path}")
                except Exception as error:
                    print(f"ERROR: Could not save the requested preview: {type(error).__name__}: {error}")
                    status = 1

            try:
                cv2.imshow("Optimized live webcam | Depth Anything V2", comparison)
                key = cv2.waitKey(1) & 0xFF
            except Exception as error:
                print(f"ERROR: Could not display live depth preview: {type(error).__name__}: {error}")
                status = 1
                break

            if elapsed >= next_report_at:
                average_ms = statistics.fmean(inference_times_ms) if inference_times_ms else 0.0
                average_age = statistics.fmean(capture_ages_ms) if capture_ages_ms else 0.0
                print(
                    f"t={elapsed:.1f}s: captured={captured_frames}, successful={successful_frames}, "
                    f"failed={inference_failures + capture_failures}, avg_inference={average_ms:.1f} ms, "
                    f"avg_FPS={current_fps:.2f}, capture_age={average_age:.1f} ms, "
                    f"camera_frames_skipped={dropped_camera_frames}"
                )
                next_report_at += 5.0

            if key in (ord("q"), ord("Q")):
                break
            if args.test_seconds is not None and elapsed >= args.test_seconds:
                break
    except Exception as error:
        print(f"ERROR: Live webcam depth test failed: {type(error).__name__}: {error}")
        status = 1
    finally:
        run_completed_at = time.perf_counter()
        if capture_worker is not None:
            capture_worker.stop()
        else:
            camera.release()
        cv2.destroyAllWindows()

    elapsed = run_completed_at - started_at
    average_inference_ms = statistics.fmean(inference_times_ms) if inference_times_ms else 0.0
    average_fps = successful_frames / elapsed if elapsed > 0 else 0.0
    average_capture_age = statistics.fmean(capture_ages_ms) if capture_ages_ms else 0.0
    print("\nOptimized live depth summary")
    print(f"Test duration: {elapsed:.1f} seconds")
    print(f"Input size: {args.input_size}; input format: {args.input_format}")
    print(f"Capture mode: {'threaded latest-frame' if capture_worker else 'synchronous'}")
    print(f"Display buffers: {'fresh' if args.fresh_buffers else 'reused'}")
    print(f"Captured frames: {captured_frames}")
    print(f"Successful depth frames: {successful_frames}")
    print(f"Failed frames: {inference_failures + capture_failures} (inference={inference_failures}, capture={capture_failures})")
    print(f"Dropped camera frames: {dropped_camera_frames}")
    print(f"Average inference time: {average_inference_ms:.1f} ms")
    print(f"Average processing FPS: {average_fps:.2f}")
    print(f"Average capture-frame age: {average_capture_age:.1f} ms")
    print(f"Depth normalization range: {normalization_range}")
    if first_inference_error is not None:
        print(f"First inference error: {first_inference_error}")
    if preview_path is not None:
        print(f"Preview snapshot: {preview_path}")

    if successful_frames == 0:
        print("ERROR: No webcam frames produced a valid depth map.")
        status = 1
    return status


if __name__ == "__main__":
    raise SystemExit(main())