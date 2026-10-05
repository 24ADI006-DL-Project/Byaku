from pathlib import Path

import cv2
from ultralytics import YOLO


WEIGHTS_PATH = Path(__file__).resolve().parent.parent / "yolo11n.pt"
CONFIDENCE_THRESHOLD = 0.25


def main() -> int:
    if not WEIGHTS_PATH.is_file():
        print(f"ERROR: YOLO weights not found: {WEIGHTS_PATH}")
        return 1

    print(f"Loading real YOLO weights: {WEIGHTS_PATH}")
    model = YOLO(str(WEIGHTS_PATH))
    print("Model loaded.")

    camera = cv2.VideoCapture(0)
    if not camera.isOpened():
        camera.release()
        print("ERROR: Could not open webcam 0.")
        return 1

    try:
        camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        success, frame = camera.read()
    finally:
        camera.release()

    if not success or frame is None:
        print("ERROR: Webcam opened, but no frame could be captured.")
        return 1

    print(f"Captured webcam frame: {frame.shape[1]}x{frame.shape[0]}")
    results = model.predict(frame, conf=CONFIDENCE_THRESHOLD, verbose=False)
    if not results or results[0].boxes is None or len(results[0].boxes) == 0:
        print("No objects detected above the confidence threshold.")
    else:
        names = model.names
        print("Detected objects:")
        for box in results[0].boxes:
            class_id = int(box.cls[0].item())
            class_name = names[class_id] if isinstance(names, (dict, list)) else str(class_id)
            confidence = float(box.conf[0].item())
            x1, y1, x2, y2 = (float(value) for value in box.xyxy[0].tolist())
            print(
                f"  {class_name}: confidence={confidence:.3f}, "
                f"bbox=({x1:.1f}, {y1:.1f}, {x2:.1f}, {y2:.1f})"
            )

    annotated_frame = results[0].plot()
    cv2.imshow("Real YOLO detection (press any key to close)", annotated_frame)
    cv2.waitKey(3000)
    cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"ERROR: {type(error).__name__}: {error}")
        raise