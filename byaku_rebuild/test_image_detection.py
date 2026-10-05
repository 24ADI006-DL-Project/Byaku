import argparse
from pathlib import Path


WEIGHTS_PATH = Path(__file__).resolve().parent.parent / "yolo11n.pt"


def main() -> int:
    parser = argparse.ArgumentParser(description="Run YOLO object detection on an image.")
    parser.add_argument("image_path", help="Path to an image file")
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.25,
        help="Minimum detection confidence from 0 to 1 (default: 0.25)",
    )
    parser.add_argument(
        "--preview-seconds",
        type=float,
        default=0,
        help="Close the annotated preview after this many seconds; 0 waits for a key",
    )
    args = parser.parse_args()
    if not 0 <= args.confidence <= 1:
        parser.error("--confidence must be between 0 and 1")
    if args.preview_seconds < 0:
        parser.error("--preview-seconds cannot be negative")

    image_path = Path(args.image_path).expanduser()
    if not image_path.is_file():
        print(f"ERROR: Image file does not exist or is not a file: {image_path}")
        return 1
    if not WEIGHTS_PATH.is_file():
        print(f"ERROR: YOLO weights not found: {WEIGHTS_PATH}")
        return 1

    try:
        import cv2
    except Exception as error:
        print(f"ERROR: Could not import OpenCV: {type(error).__name__}: {error}")
        return 1

    image = cv2.imread(str(image_path))
    if image is None:
        print(f"ERROR: Could not read image: {image_path}")
        return 1

    try:
        from ultralytics import YOLO

        model = YOLO(str(WEIGHTS_PATH))
        results = model.predict(image, conf=args.confidence, verbose=False)
    except Exception as error:
        print(f"ERROR: YOLO model or inference failed: {type(error).__name__}: {error}")
        return 1

    if not results:
        print("ERROR: YOLO returned no results for the image.")
        return 1

    result = results[0]
    boxes = result.boxes
    print(f"Image: {image_path} ({image.shape[1]}x{image.shape[0]})")
    print(f"Confidence threshold: {args.confidence:.2f}")
    if boxes is None or len(boxes) == 0:
        print("No objects detected above the confidence threshold.")
    else:
        names = model.names
        print("Detected objects:")
        for box in boxes:
            class_id = int(box.cls[0].item())
            class_name = names[class_id] if isinstance(names, (dict, list)) else str(class_id)
            confidence = float(box.conf[0].item())
            x1, y1, x2, y2 = (float(value) for value in box.xyxy[0].tolist())
            print(
                f"  {class_name}: confidence={confidence:.3f}, "
                f"bbox=({x1:.1f}, {y1:.1f}, {x2:.1f}, {y2:.1f})"
            )

    annotated_image = result.plot(labels=True, conf=True)
    try:
        cv2.imshow("YOLO image detection (press any key to close)", annotated_image)
        wait_ms = 0 if args.preview_seconds == 0 else max(1, int(args.preview_seconds * 1000))
        cv2.waitKey(wait_ms)
    except Exception as error:
        print(f"ERROR: Could not display annotated image: {type(error).__name__}: {error}")
        return 1
    finally:
        cv2.destroyAllWindows()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())