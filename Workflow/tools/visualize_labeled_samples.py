from __future__ import annotations

import argparse
import random
from pathlib import Path

import cv2

from Workflow.tools.dataset_validation import (
    CLASS_NAMES,
    IMAGE_EXTENSIONS,
    SPLITS,
    parse_yolo_label,
    validate_dataset,
)


def _choose_samples(records: list[dict], count: int, seed: int) -> list[dict]:
    if count < 1:
        raise ValueError("sample count must be positive")
    by_class: dict[int, list[dict]] = {}
    for record in records:
        for class_id in {box["class_id"] for box in record["boxes"]}:
            by_class.setdefault(class_id, []).append(record)
    chooser = random.Random(seed)
    selected: list[dict] = []
    for class_id in sorted(by_class):
        choices = [record for record in by_class[class_id] if record not in selected]
        if choices and len(selected) < count:
            selected.append(chooser.choice(choices))
    remaining = [record for record in records if record not in selected]
    chooser.shuffle(remaining)
    selected.extend(remaining[: max(0, count - len(selected))])
    return selected


def visualize_samples(
    dataset_root: Path,
    *,
    count: int = 16,
    seed: int = 39,
    output_dir: Path | None = None,
) -> dict:
    root = dataset_root.resolve()
    report = validate_dataset(root)["quality_report"]
    records: list[dict] = []
    for split in SPLITS:
        image_dir = root / "images" / split
        if not image_dir.is_dir():
            continue
        for image_path in sorted(image_dir.rglob("*")):
            if not image_path.is_file() or image_path.suffix.lower() not in IMAGE_EXTENSIONS:
                continue
            label_path = (root / "labels" / split / image_path.relative_to(image_dir)).with_suffix(".txt")
            if not label_path.is_file():
                continue
            try:
                boxes = parse_yolo_label(label_path.read_text(encoding="utf-8"))
            except (OSError, UnicodeError, ValueError):
                continue
            if boxes:
                records.append(
                    {
                        "image_path": image_path,
                        "label_path": label_path,
                        "boxes": boxes,
                    }
                )
    selected = _choose_samples(records, count, seed) if records else []
    destination_root = (output_dir or root / "metadata" / "qa_samples").resolve()
    saved: list[str] = []
    class_samples = {
        class_name: sum(
            any(box["class_id"] == class_id for box in record["boxes"])
            for record in selected
        )
        for class_id, class_name in enumerate(CLASS_NAMES)
    }
    if selected:
        destination_root.mkdir(parents=True, exist_ok=True)
    for index, record in enumerate(selected, start=1):
        image = cv2.imread(str(record["image_path"]), cv2.IMREAD_COLOR)
        if image is None or image.size == 0:
            continue
        height, width = image.shape[:2]
        for box in record["boxes"]:
            x_center = box["x_center"] * width
            y_center = box["y_center"] * height
            box_width = box["width"] * width
            box_height = box["height"] * height
            x1 = round(x_center - box_width / 2)
            y1 = round(y_center - box_height / 2)
            x2 = round(x_center + box_width / 2)
            y2 = round(y_center + box_height / 2)
            class_name = CLASS_NAMES[box["class_id"]]
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 200, 0), 2)
            cv2.putText(
                image,
                class_name,
                (x1, max(18, y1 - 6)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 200, 0),
                2,
                cv2.LINE_AA,
            )
        output_path = destination_root / f"qa_{index:03d}_{record['image_path'].name}"
        if not cv2.imwrite(str(output_path), image):
            raise OSError(f"could not save QA visualization: {output_path}")
        saved.append(str(output_path))
    return {
        "requested_count": count,
        "available_annotated_samples": len(records),
        "saved_count": len(saved),
        "saved_images": saved,
        "class_samples": class_samples,
        "classes_without_valid_annotations": [
            name for name, sample_count in class_samples.items() if sample_count == 0
        ],
        "dataset_valid": report["valid"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Create read-only visual QA overlays for annotated samples.")
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "Data" / "datasets" / "byaku_detection",
    )
    parser.add_argument("--count", type=int, default=16)
    parser.add_argument("--seed", type=int, default=39)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    result = visualize_samples(
        args.dataset_root,
        count=args.count,
        seed=args.seed,
        output_dir=args.output_dir,
    )
    if not result["available_annotated_samples"]:
        print("No valid annotated samples are available; no QA images were fabricated.")
    else:
        print(
            f"Saved {result['saved_count']} of {result['available_annotated_samples']} "
            f"annotated samples to {args.output_dir or args.dataset_root / 'metadata' / 'qa_samples'}"
        )
        print("Classes without valid annotations:", ", ".join(result["classes_without_valid_annotations"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
