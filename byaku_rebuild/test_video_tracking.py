import argparse
import math
from pathlib import Path
import time


WEIGHTS_PATH = Path(__file__).resolve().parent.parent / "yolo11n.pt"


def intersection_over_union(first_box: tuple[float, float, float, float], second_box: tuple[float, float, float, float]) -> float:
    left = max(first_box[0], second_box[0])
    top = max(first_box[1], second_box[1])
    right = min(first_box[2], second_box[2])
    bottom = min(first_box[3], second_box[3])
    intersection = max(0.0, right - left) * max(0.0, bottom - top)
    first_area = max(0.0, first_box[2] - first_box[0]) * max(0.0, first_box[3] - first_box[1])
    second_area = max(0.0, second_box[2] - second_box[0]) * max(0.0, second_box[3] - second_box[1])
    union = first_area + second_area - intersection
    return intersection / union if union > 0 else 0.0


def track_color(track_id: int) -> tuple[int, int, int]:
    return (
        64 + (track_id * 67) % 192,
        64 + (track_id * 113) % 192,
        64 + (track_id * 157) % 192,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Run YOLO and ByteTrack on a video.")
    parser.add_argument("video_path", help="Path to the source video")
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.25,
        help="Minimum detection confidence from 0 to 1 (default: 0.25)",
    )
    args = parser.parse_args()
    if not 0 <= args.confidence <= 1:
        parser.error("--confidence must be between 0 and 1")

    video_path = Path(args.video_path).expanduser()
    if not video_path.is_file():
        print(f"ERROR: Video file does not exist or is not a file: {video_path}")
        return 1
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

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        capture.release()
        print(f"ERROR: Could not open video: {video_path}")
        return 1

    try:
        print(f"Loading YOLO weights: {WEIGHTS_PATH}")
        model = YOLO(str(WEIGHTS_PATH))
    except Exception as error:
        capture.release()
        print(f"ERROR: Could not load YOLO model: {type(error).__name__}: {error}")
        return 1

    source_fps = float(capture.get(cv2.CAP_PROP_FPS))
    reported_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    print(
        f"Video: {video_path} ({int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))}x"
        f"{int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))}, {source_fps:.2f} FPS, "
        f"{reported_frames} reported frames)"
    )
    print(f"Confidence threshold: {args.confidence:.2f}; tracker: Ultralytics ByteTrack")
    print("Playback running. Press q in the video window to stop.")

    frame_number = 0
    tracked_detections = 0
    untracked_detections = 0
    frames_with_tracks = 0
    detected_classes: set[str] = set()
    previous_tracks: dict[int, tuple[str, tuple[float, float, float, float]]] = {}
    track_history: dict[int, dict[str, int | str]] = {}
    consecutive_id_pairs = 0
    moving_id_pairs = 0
    lost_track_events: list[tuple[int, int]] = []
    possible_id_replacements: list[tuple[int, int, int, float]] = []
    duplicate_track_pairs: list[tuple[int, int, int, float]] = []
    status = 0

    try:
        while True:
            frame_started = time.perf_counter()
            success, frame = capture.read()
            if not success:
                break

            frame_number += 1
            try:
                results = model.track(
                    frame,
                    conf=args.confidence,
                    persist=True,
                    tracker="bytetrack.yaml",
                    verbose=False,
                )
            except Exception as error:
                print(f"ERROR: YOLO/ByteTrack failed on frame {frame_number}: {type(error).__name__}: {error}")
                status = 1
                break

            if not results:
                print(f"ERROR: YOLO/ByteTrack returned no result for frame {frame_number}.")
                status = 1
                break

            result = results[0]
            boxes = result.boxes
            current_tracks: dict[int, tuple[str, tuple[float, float, float, float]]] = {}
            current_detections: list[tuple[int | None, str, float, tuple[float, float, float, float]]] = []
            if boxes is not None:
                names = model.names
                for index in range(len(boxes)):
                    class_id = int(boxes.cls[index].item())
                    class_name = names[class_id] if isinstance(names, (dict, list)) else str(class_id)
                    confidence = float(boxes.conf[index].item())
                    coordinates = tuple(float(value) for value in boxes.xyxy[index].tolist())
                    track_id = int(boxes.id[index].item()) if boxes.id is not None else None
                    current_detections.append((track_id, class_name, confidence, coordinates))
                    detected_classes.add(class_name)
                    if track_id is None:
                        untracked_detections += 1
                        print(
                            f"Frame {frame_number}: {class_name} | id=untracked | "
                            f"confidence={confidence:.3f} | bbox={coordinates}"
                        )
                    else:
                        tracked_detections += 1
                        current_tracks[track_id] = (class_name, coordinates)
                        print(
                            f"Frame {frame_number}: {class_name} | id={track_id} | "
                            f"confidence={confidence:.3f} | bbox={coordinates}"
                        )

            if current_tracks:
                frames_with_tracks += 1

            for track_id, (class_name, coordinates) in current_tracks.items():
                history = track_history.setdefault(
                    track_id,
                    {"class_name": class_name, "first_frame": frame_number, "last_frame": frame_number,
                     "observations": 0, "consecutive_pairs": 0, "moving_pairs": 0},
                )
                history["last_frame"] = frame_number
                history["observations"] = int(history["observations"]) + 1
                previous = previous_tracks.get(track_id)
                if previous is not None and previous[0] == class_name:
                    consecutive_id_pairs += 1
                    history["consecutive_pairs"] = int(history["consecutive_pairs"]) + 1
                    old_box = previous[1]
                    old_center = ((old_box[0] + old_box[2]) / 2, (old_box[1] + old_box[3]) / 2)
                    new_center = ((coordinates[0] + coordinates[2]) / 2, (coordinates[1] + coordinates[3]) / 2)
                    if math.dist(old_center, new_center) > 1.0:
                        moving_id_pairs += 1
                        history["moving_pairs"] = int(history["moving_pairs"]) + 1

            lost_ids = set(previous_tracks) - set(current_tracks)
            new_ids = set(current_tracks) - set(previous_tracks)
            for track_id in sorted(lost_ids):
                lost_track_events.append((frame_number, track_id))
                print(f"Frame {frame_number}: track id={track_id} no longer present (lost/occluded/left frame).")

            for new_id in sorted(new_ids):
                new_class, new_box = current_tracks[new_id]
                for old_id in sorted(lost_ids):
                    old_class, old_box = previous_tracks[old_id]
                    if old_class == new_class:
                        overlap = intersection_over_union(old_box, new_box)
                        if overlap >= 0.30:
                            possible_id_replacements.append((frame_number, old_id, new_id, overlap))
                            print(
                                f"Frame {frame_number}: possible ID replacement {old_id}->{new_id} "
                                f"for {new_class} (IoU={overlap:.2f})."
                            )

            current_items = list(current_tracks.items())
            for first_index, (first_id, (first_class, first_box)) in enumerate(current_items):
                for second_id, (second_class, second_box) in current_items[first_index + 1:]:
                    if first_class == second_class:
                        overlap = intersection_over_union(first_box, second_box)
                        if overlap >= 0.85:
                            duplicate_track_pairs.append((frame_number, first_id, second_id, overlap))
                            print(
                                f"Frame {frame_number}: possible duplicate {first_class} tracks "
                                f"{first_id} and {second_id} (IoU={overlap:.2f})."
                            )

            annotated_frame = frame.copy()
            for track_id, class_name, confidence, coordinates in current_detections:
                x1, y1, x2, y2 = (int(round(value)) for value in coordinates)
                color = track_color(track_id) if track_id is not None else (128, 128, 128)
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
                track_label = str(track_id) if track_id is not None else "untracked"
                label = f"{class_name} ID:{track_label} {confidence:.2f}"
                text_y = max(18, y1 - 7)
                cv2.putText(
                    annotated_frame,
                    label,
                    (x1, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    color,
                    2,
                    cv2.LINE_AA,
                )

            elapsed = time.perf_counter() - frame_started
            display_fps = 1.0 / elapsed if elapsed > 0 else 0.0
            cv2.putText(
                annotated_frame,
                f"FPS: {display_fps:.1f} | Frame: {frame_number}",
                (12, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )
            cv2.imshow("YOLO + ByteTrack video test (press q to stop)", annotated_frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

            previous_tracks = current_tracks
    except Exception as error:
        print(f"ERROR: Video tracking failed: {type(error).__name__}: {error}")
        status = 1
    finally:
        capture.release()
        cv2.destroyAllWindows()

    if frame_number == 0 and status == 0:
        print(f"ERROR: Video opened but no frames could be read: {video_path}")
        status = 1

    classes = ", ".join(sorted(detected_classes)) if detected_classes else "none"
    print("\nTracking summary")
    print(f"Frames processed: {frame_number}")
    print(f"Tracked detections: {tracked_detections}")
    print(f"Untracked detections: {untracked_detections}")
    print(f"Frames with tracked objects: {frames_with_tracks}")
    print(f"Classes detected: {classes}")
    print(f"Consecutive same-ID frame pairs: {consecutive_id_pairs}")
    print(f"Same-ID pairs with bbox movement: {moving_id_pairs}")
    print(f"Lost-track events: {len(lost_track_events)}")
    print(f"Possible ID replacements: {len(possible_id_replacements)}")
    print(f"Possible duplicate-track pairs: {len(duplicate_track_pairs)}")

    examples = [
        (track_id, history)
        for track_id, history in track_history.items()
        if int(history["consecutive_pairs"]) > 0
    ]
    if examples:
        print("Track continuity examples:")
        for track_id, history in examples[:8]:
            print(
                f"  {history['class_name']} id={track_id}: frames {history['first_frame']}-"
                f"{history['last_frame']}, observations={history['observations']}, "
                f"consecutive_pairs={history['consecutive_pairs']}, "
                f"moving_pairs={history['moving_pairs']}"
            )
    else:
        print("Track continuity examples: none; no ID persisted across consecutive frames.")

    return status


if __name__ == "__main__":
    raise SystemExit(main())