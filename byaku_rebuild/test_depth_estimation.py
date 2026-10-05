import argparse
from pathlib import Path
import time


MODEL_NAME = "depth-anything/Depth-Anything-V2-Small-hf"
RESULTS_DIR = Path(__file__).resolve().parent / "results"


def next_output_path(image_path: Path) -> Path:
    base_path = RESULTS_DIR / f"{image_path.stem}_depth.png"
    if not base_path.exists():
        return base_path

    version = 2
    while True:
        candidate = RESULTS_DIR / f"{image_path.stem}_depth_{version}.png"
        if not candidate.exists():
            return candidate
        version += 1


def main() -> int:
    parser = argparse.ArgumentParser(description="Estimate relative depth for one image.")
    parser.add_argument("image_path", help="Path to an input image")
    parser.add_argument(
        "--preview-seconds",
        type=float,
        default=0,
        help="Close the side-by-side preview after this many seconds; 0 waits for a key",
    )
    args = parser.parse_args()
    if args.preview_seconds < 0:
        parser.error("--preview-seconds cannot be negative")

    image_path = Path(args.image_path).expanduser()
    if not image_path.is_file():
        print(f"ERROR: Image file does not exist or is not a file: {image_path}")
        return 1

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

    image = cv2.imread(str(image_path))
    if image is None:
        print(f"ERROR: Could not read image: {image_path}")
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

    height, width = image.shape[:2]
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    try:
        inputs = processor(images=Image.fromarray(rgb_image), return_tensors="pt")
        inputs = {name: value.to(device) for name, value in inputs.items()}
        started_at = time.perf_counter()
        with torch.inference_mode():
            prediction = model(**inputs).predicted_depth
            prediction = functional.interpolate(
                prediction.unsqueeze(1),
                size=(height, width),
                mode="bicubic",
                align_corners=False,
            ).squeeze(0).squeeze(0)
        inference_seconds = time.perf_counter() - started_at
        depth_values = prediction.detach().cpu().numpy().astype(np.float32)
    except Exception as error:
        print(f"ERROR: Depth inference failed: {type(error).__name__}: {error}")
        return 1

    finite = np.isfinite(depth_values)
    if not finite.any():
        print("ERROR: Model produced no finite depth values.")
        return 1

    minimum = float(depth_values[finite].min())
    maximum = float(depth_values[finite].max())
    normalized = np.zeros(depth_values.shape, dtype=np.float32)
    if maximum > minimum:
        normalized[finite] = (depth_values[finite] - minimum) / (maximum - minimum)
    depth_visualization = cv2.applyColorMap(
        np.clip(normalized * 255, 0, 255).astype(np.uint8),
        cv2.COLORMAP_INFERNO,
    )

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = next_output_path(image_path)
    try:
        if not cv2.imwrite(str(output_path), depth_visualization):
            print(f"ERROR: Could not save depth visualization: {output_path}")
            return 1
    except Exception as error:
        print(f"ERROR: Could not save depth visualization: {type(error).__name__}: {error}")
        return 1

    print(f"Input image: {image_path}")
    print(f"Image dimensions: {width}x{height}")
    print(f"Model: {MODEL_NAME}")
    print(f"Device: {device}")
    print(f"Inference time: {inference_seconds:.3f} seconds")
    print(f"Depth visualization: {output_path}")
    print("Depth values are normalized relative estimates, not calibrated distances in meters.")

    caption_height = 36
    original_panel = image.copy()
    depth_panel = depth_visualization.copy()
    cv2.rectangle(original_panel, (0, 0), (width, caption_height), (32, 32, 32), -1)
    cv2.rectangle(depth_panel, (0, 0), (width, caption_height), (32, 32, 32), -1)
    cv2.putText(original_panel, "Original", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(depth_panel, "Relative depth (Inferno)", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    comparison = np.concatenate((original_panel, depth_panel), axis=1)
    preview_scale = min(1.0, 1600 / comparison.shape[1], 900 / comparison.shape[0])
    if preview_scale < 1.0:
        comparison = cv2.resize(
            comparison,
            (int(comparison.shape[1] * preview_scale), int(comparison.shape[0] * preview_scale)),
            interpolation=cv2.INTER_AREA,
        )

    try:
        cv2.imshow("Original image | Depth Anything V2 relative depth", comparison)
        if args.preview_seconds == 0:
            print("Press any key in the preview window to close it.")
            cv2.waitKey(0)
        else:
            cv2.waitKey(max(1, int(args.preview_seconds * 1000)))
    except Exception as error:
        print(f"ERROR: Could not display comparison image: {type(error).__name__}: {error}")
        return 1
    finally:
        cv2.destroyAllWindows()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())