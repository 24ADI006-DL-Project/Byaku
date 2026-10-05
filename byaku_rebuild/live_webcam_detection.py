import argparse
import time
from pathlib import Path


WEIGHTS_PATH = Path(__file__).resolve().parent.parent / "yolo11n.pt"
CONFIDENCE_THRESHOLD = 0.25


def main() -> int:
    parser = argparse.ArgumentParser(description="Run live YOLO detection on webcam 0.")
    parser.add_argument(
        "--test-seconds",
        type=float,
        help="Stop automatically after this many seconds (useful for a short smoke test).",
    )
    args = parser.parse_args()
    if args.test_seconds is not None and args.test_seconds <= 0:
        parser.error("--test-seconds must be greater than zero")

    if not WEIGHTS_PATH.is_file():
        print(f"ERROR: YOLO weights not found: {WEIGHTS_PATH}")
        return 1

    try:
        import cv2
    except Exception as error:
        print(f"ERROR: Could not import OpenCV: {type(error).__name__}: {error}")
        return 1

    try:
        from ultralytics import YOLO
    except Exception as error:
        print(f"ERROR: Could not import Ultralytics YOLO: {type(error).__name__}: {error}")
        return 1

    try:
        print(f"Loading YOLO weights: {WEIGHTS_PATH}")
        model = YOLO(str(WEIGHTS_PATH))
    except Exception as error:
        print(f"ERROR: Could not load YOLO model: {type(error).__name__}: {error}")
        return 1

    try:
        camera = cv2.VideoCapture(0)
    except Exception as error:
        print(f"ERROR: Could not initialize webcam 0: {type(error).__name__}: {error}")
        return 1

    if not camera.isOpened():
        camera.release()
        print("ERROR: Could not open webcam 0. Check that it is connected and available.")
        return 1

    frame_count = 0
    frames_with_detections = 0
    total_detections = 0
    classes_seen: set[str] = set()
    status = 0
    started_at = time.perf_counter()
    print("Live detection running. Press q in the video window to exit.")

    try:
        camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        while True:
            loop_started_at = time.perf_counter()
            success, frame = camera.read()
            if not success or frame is None:
                print("ERROR: Webcam opened, but a frame could not be read.")
                status = 1
                break

            try:
                results = model.predict(frame, conf=CONFIDENCE_THRESHOLD, verbose=False)
            except Exception as error:
                print(f"ERROR: YOLO inference failed: {type(error).__name__}: {error}")
                status = 1
                break

            if not results:
                print("ERROR: YOLO returned no result for the webcam frame.")
                status = 1
                break

            result = results[0]
            boxes = result.boxes
            if boxes is not None and len(boxes) > 0:
                frames_with_detections += 1
                total_detections += len(boxes)
                names = model.names
                for class_id in boxes.cls.tolist():
                    class_id = int(class_id)
                    class_name = names[class_id] if isinstance(names, (dict, list)) else str(class_id)
                    classes_seen.add(class_name)

            frame_count += 1
            elapsed = time.perf_counter() - loop_started_at
            fps = 1.0 / elapsed if elapsed > 0 else 0.0
            annotated_frame = result.plot(labels=True, conf=True)
            cv2.putText(
                annotated_frame,
                f"FPS: {fps:.1f}",
                (12, 32),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )
            cv2.imshow("Live YOLO detection (press q to exit)", annotated_frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
            if args.test_seconds is not None and time.perf_counter() - started_at >= args.test_seconds:
                break
    except Exception as error:
        print(f"ERROR: Live webcam detection failed: {type(error).__name__}: {error}")
        status = 1
    finally:
        camera.release()
        cv2.destroyAllWindows()

    if status == 0:
        classes = ", ".join(sorted(classes_seen)) if classes_seen else "none"
        print(
            f"Stopped after {frame_count} frames; detections on {frames_with_detections} frames, "
            f"{total_detections} total detections; classes: {classes}."
        )
    return status


if __name__ == "__main__":
    raise SystemExit(main())