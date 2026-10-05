import argparse
from pathlib import Path
import statistics
import time


MODEL_NAME = "depth-anything/Depth-Anything-V2-Small-hf"
PREVIEW_DIR = Path(__file__).resolve().parent / "results"


def unique_preview_path() -> Path:
    path = PREVIEW_DIR / "live_webcam_depth_preview.png"
    if not path.exists():
        return path

    version = 2
    while True:
        candidate = PREVIEW_DIR / f"live_webcam_depth_preview_{version}.png"
        if not candidate.exists():
            return candidate
        version += 1


def main() -> int:
    parser = argparse.ArgumentParser(description="Run live Depth Anything V2 on the default webcam.")
    parser.add_argument(
        "--test-seconds",
        type=float,
        help="Stop automatically after this many seconds; otherwise run until Q is pressed",
    )
    parser.add_argument(
        "--save-preview",
        action="store_true",
        help="Save one original/depth comparison frame under byaku_rebuild/results/",
    )
    args = parser.parse_args()
    if args.test_seconds is not None and args.test_seconds <= 0:
        parser.error("--test-seconds must be greater than zero")

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

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
    normalization_range: tuple[float, float] | None = None
    inference_times_ms: list[float] = []
    captured_frames = 0
    successful_frames = 0
    inference_failures = 0
    capture_failures = 0
    first_inference_error: str | None = None
    preview_path: Path | None = None
    started_at = time.perf_counter()
    next_report_at = 5.0
    status = 0
    print(f"Model: {MODEL_NAME}; device: {device}; local cache only.")
    print("Live depth running. Press Q in the preview window to stop.")

    try:
        while True:
            capture_ok, frame = camera.read()
            if not capture_ok or frame is None:
                capture_failures += 1
                print("ERROR: Webcam stopped returning readable frames.")
                status = 1
                break

            captured_frames += 1
            height, width = frame.shape[:2]
            frame_started_at = time.perf_counter()
            try:
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                inputs = processor(images=Image.fromarray(rgb_frame), return_tensors="pt")
                inputs = {name: value.to(device) for name, value in inputs.items()}
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                with torch.inference_mode():
                    prediction = model(**inputs).predicted_depth
                    prediction = functional.interpolate(
                        prediction.unsqueeze(1),
                        size=(height, width),
                        mode="bicubic",
                        align_corners=False,
                    ).squeeze(0).squeeze(0)
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                depth_values = prediction.detach().cpu().numpy().astype(np.float32)
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
                clipped_depth = np.clip(
                    (depth_values - normalization_range[0])
                    / (normalization_range[1] - normalization_range[0]),
                    0.0,
                    1.0,
                )
                depth_visualization = cv2.applyColorMap(
                    (clipped_depth * 255).astype(np.uint8),
                    cv2.COLORMAP_INFERNO,
                )
                successful_frames += 1
                frame_error = None
            except Exception as error:
                inference_failures += 1
                frame_error = f"{type(error).__name__}: {error}"
                if first_inference_error is None:
                    first_inference_error = frame_error
                    print(f"ERROR: Depth inference failed on frame {captured_frames}: {frame_error}")
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

            original_panel = frame.copy()
            depth_panel = depth_visualization.copy()
            elapsed = time.perf_counter() - started_at
            current_fps = successful_frames / elapsed if elapsed > 0 else 0.0
            displayed_inference_ms = inference_times_ms[-1] if inference_times_ms else 0.0
            cv2.putText(
                original_panel,
                f"Inference: {displayed_inference_ms:.0f} ms | FPS: {current_fps:.2f}",
                (10, 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                depth_panel,
                "Relative depth (fixed scale)",
                (10, 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            comparison = np.concatenate((original_panel, depth_panel), axis=1)
            display_scale = min(1.0, 1600 / comparison.shape[1], 900 / comparison.shape[0])
            if display_scale < 1.0:
                comparison = cv2.resize(
                    comparison,
                    (int(comparison.shape[1] * display_scale), int(comparison.shape[0] * display_scale)),
                    interpolation=cv2.INTER_AREA,
                )

            if args.save_preview and preview_path is None and frame_error is None:
                try:
                    PREVIEW_DIR.mkdir(parents=True, exist_ok=True)
                    preview_path = unique_preview_path()
                    if not cv2.imwrite(str(preview_path), comparison):
                        raise OSError(f"OpenCV could not write {preview_path}")
                    print(f"Single preview snapshot: {preview_path}")
                except Exception as error:
                    print(f"ERROR: Could not save the requested preview snapshot: {type(error).__name__}: {error}")
                    status = 1

            try:
                cv2.imshow("Live webcam | Depth Anything V2", comparison)
                key = cv2.waitKey(1) & 0xFF
            except Exception as error:
                print(f"ERROR: Could not display live depth preview: {type(error).__name__}: {error}")
                status = 1
                break

            if elapsed >= next_report_at:
                average_inference_ms = statistics.fmean(inference_times_ms) if inference_times_ms else 0.0
                print(
                    f"t={elapsed:.1f}s: captured={captured_frames}, successful={successful_frames}, "
                    f"failed={inference_failures + capture_failures}, "
                    f"avg_inference={average_inference_ms:.1f} ms, avg_FPS={current_fps:.2f}"
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
        camera.release()
        cv2.destroyAllWindows()

    elapsed = time.perf_counter() - started_at
    average_inference_ms = statistics.fmean(inference_times_ms) if inference_times_ms else 0.0
    average_fps = successful_frames / elapsed if elapsed > 0 else 0.0
    failed_frames = inference_failures + capture_failures
    print("\nLive depth test summary")
    print(f"Test duration: {elapsed:.1f} seconds")
    print(f"Captured frames: {captured_frames}")
    print(f"Successful depth frames: {successful_frames}")
    print(f"Failed frames: {failed_frames} (inference={inference_failures}, capture={capture_failures})")
    print(f"Average inference time: {average_inference_ms:.1f} ms")
    print(f"Average processing FPS: {average_fps:.2f}")
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