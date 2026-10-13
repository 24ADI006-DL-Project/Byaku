from __future__ import annotations

import argparse
from collections import Counter
from contextlib import redirect_stderr, redirect_stdout
from datetime import datetime
import csv
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import sys
import time
import traceback
from typing import Any, TextIO

import cv2
import numpy as np
from PIL import Image
import psutil
import yaml

from Workflow.training.prepare_open_images_training import (
    WORKSPACE_ROOT,
    DEFAULT_OUTPUT_ROOT,
    TRAINABLE_CLASSES,
)
from Workflow.tools.dataset_validation import parse_yolo_label


BASE_CHECKPOINT = WORKSPACE_ROOT / "Models" / "yolo11n.pt"
DEFAULT_PROJECT = WORKSPACE_ROOT / "Models" / "training_runs"
IMAGE_SIZE = 640
EPOCHS = 20
PATIENCE = 5
BATCH_SIZE = 16
WORKERS = 2
SEED = 42
VALIDATION_CONFIDENCE = 0.001
VALIDATION_IOU = 0.7
MAX_DETECTIONS = 300
COCO_CLASS_BY_TARGET_ID = {
    0: 0,
    1: 39,
    2: 73,
    3: 56,
    4: 60,
    5: 67,
    6: 63,
}
TARGET_ID_BY_COCO_CLASS = {
    source_id: target_id
    for target_id, source_id in COCO_CLASS_BY_TARGET_ID.items()
}
TARGET_NAMES = {entry["target_id"]: entry["name"] for entry in TRAINABLE_CLASSES}
ESTIMATED_PRIOR_TRAIN_SECONDS = 5075.32
ESTIMATED_PRIOR_EPOCHS = 19


class _Tee(io.TextIOBase):
    def __init__(self, *streams: TextIO) -> None:
        self._streams = streams

    def write(self, text: str) -> int:
        for stream in self._streams:
            stream.write(text)
        return len(text)

    def flush(self) -> None:
        for stream in self._streams:
            stream.flush()

    def isatty(self) -> bool:
        return any(stream.isatty() for stream in self._streams)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_write(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


def _validate_dataset(data_yaml: Path) -> tuple[dict[str, Any], dict[str, int]]:
    config = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))
    if not isinstance(config, dict) or not isinstance(config.get("names"), dict):
        raise ValueError(f"dataset YAML is missing class names: {data_yaml}")
    names = {int(key): str(value) for key, value in config["names"].items()}
    if names != TARGET_NAMES:
        raise ValueError(f"training must use the audited seven-class map, got {names!r}")
    dataset_root = Path(config["path"]).resolve()
    counts: dict[str, Counter[int]] = {}
    image_counts: dict[str, int] = {}
    image_hashes: dict[str, str] = {}
    for split in ("train", "val", "test"):
        image_dir = dataset_root / config[split]
        label_dir = dataset_root / "labels" / split
        images = sorted(
            path for path in image_dir.iterdir()
            if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png"}
        )
        if not images:
            raise ValueError(f"{split} split contains no images")
        split_box_counts: Counter[int] = Counter()
        seen = set()
        for image_path in images:
            label_path = label_dir / f"{image_path.stem}.txt"
            if not label_path.is_file():
                raise ValueError(f"missing {split} label for {image_path.name}")
            with Image.open(image_path) as image:
                image.verify()
            label_text = label_path.read_text(encoding="utf-8")
            parsed = parse_yolo_label(label_text)
            for item in parsed:
                split_box_counts[item["class_id"]] += 1
            digest = _sha256(image_path)
            previous_split = image_hashes.get(digest)
            if previous_split is not None and previous_split != split:
                raise ValueError(f"exact duplicate image leakage between {previous_split} and {split}")
            image_hashes[digest] = split
            seen.add(image_path.stem)
        orphan_labels = {
            label.stem for label in label_dir.glob("*.txt")
        } - seen
        if orphan_labels:
            raise ValueError(f"orphan {split} labels: {sorted(orphan_labels)[:5]}")
        counts[split] = split_box_counts
        image_counts[split] = len(images)
    if any(counts["train"][class_id] == 0 for class_id in TARGET_NAMES):
        raise ValueError("train split is missing boxes for at least one trainable target class")
    return config, {
        f"{split}_{TARGET_NAMES[class_id]}": counts[split][class_id]
        for split in ("train", "val", "test")
        for class_id in TARGET_NAMES
    } | {f"{split}_images": image_counts[split] for split in image_counts}


def run_training(
    *,
    name: str | None = None,
    data_yaml: Path = DEFAULT_OUTPUT_ROOT / "data.yaml",
    base_checkpoint: Path = BASE_CHECKPOINT,
    project: Path = DEFAULT_PROJECT,
) -> dict[str, Any]:
    data_yaml = Path(data_yaml).resolve()
    base_checkpoint = Path(base_checkpoint).resolve()
    project = Path(project).resolve()
    if not data_yaml.is_file():
        raise FileNotFoundError(f"prepared dataset YAML not found: {data_yaml}")
    if not base_checkpoint.is_file():
        raise FileNotFoundError(f"required original checkpoint not found: {base_checkpoint}")
    try:
        project.relative_to((WORKSPACE_ROOT / "Models").resolve())
    except ValueError as error:
        raise ValueError(f"training output must remain inside {WORKSPACE_ROOT / 'Models'}") from error
    _, split_counts = _validate_dataset(data_yaml)

    name = name or f"yolo11n_byaku_openimages7_{datetime.now():%Y%m%d_%H%M%S}"
    run_dir = project / name
    if run_dir.exists():
        raise FileExistsError(f"refusing to overwrite existing training run: {run_dir}")
    project.mkdir(parents=True, exist_ok=True)
    free_bytes = shutil.disk_usage(project).free
    if free_bytes < 10 * 1024**3:
        raise RuntimeError(
            f"insufficient free space for a new training run: {free_bytes / 1024**3:.2f} GiB"
        )
    run_dir.mkdir(parents=True)
    checkpoint_hash_before = _sha256(base_checkpoint)
    available_ram_gib = psutil.virtual_memory().available / 1024**3
    estimate_seconds = (
        ESTIMATED_PRIOR_TRAIN_SECONDS
        / ESTIMATED_PRIOR_EPOCHS
        * EPOCHS
        * (16 / BATCH_SIZE)
    )
    settings = {
        "model": str(base_checkpoint),
        "model_sha256_before": checkpoint_hash_before,
        "data": str(data_yaml),
        "data_split_counts": split_counts,
        "epochs": EPOCHS,
        "patience": PATIENCE,
        "batch": BATCH_SIZE,
        "imgsz": IMAGE_SIZE,
        "device": "cpu",
        "workers": WORKERS,
        "seed": SEED,
        "deterministic": True,
        "fraction": 1.0,
        "train_split": "train only",
        "validation_split": "val only (early stopping)",
        "test_split": "not used during training",
        "optimizer": "auto",
        "project": str(project),
        "name": name,
        "exist_ok": True,
        "pretrained": True,
        "validation_confidence": VALIDATION_CONFIDENCE,
        "validation_iou": VALIDATION_IOU,
        "max_det": MAX_DETECTIONS,
        "available_ram_gib_before_training": round(available_ram_gib, 2),
        "free_disk_gib_before_training": round(free_bytes / 1024**3, 2),
        "estimated_duration_seconds": round(estimate_seconds, 1),
        "estimate_basis": (
            "Prior run recorded 5,075.32 seconds through epoch 19 on the same dataset scale "
            "at batch 16; extrapolated linearly to 20 epochs at batch 16."
        ),
    }
    _json_write(run_dir / "run_UI_config.json", settings)
    started = time.perf_counter()
    summary: dict[str, Any] = {
        "status": "running",
        "run_dir": str(run_dir),
        "config": settings,
    }
    log_path = run_dir / "training.log"
    try:
        config_dir = WORKSPACE_ROOT / "Config" / "ultralytics"
        config_dir.mkdir(parents=True, exist_ok=True)
        os.environ["YOLO_CONFIG_DIR"] = str(config_dir)
        os.environ["MPLCONFIGDIR"] = str(config_dir / "matplotlib")
        Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
        from ultralytics import YOLO

        with log_path.open("w", encoding="utf-8", buffering=1) as log:
            tee = _Tee(sys.stdout, log)
            error_tee = _Tee(sys.stderr, log)
            with redirect_stdout(tee), redirect_stderr(error_tee):
                print(f"Training run: {run_dir}")
                print(f"Starting from original checkpoint: {base_checkpoint}")
                model = YOLO(str(base_checkpoint))
                results = model.train(
                    data=str(data_yaml),
                    epochs=EPOCHS,
                    patience=PATIENCE,
                    batch=BATCH_SIZE,
                    imgsz=IMAGE_SIZE,
                    device="cpu",
                    workers=WORKERS,
                    seed=SEED,
                    deterministic=True,
                    fraction=1.0,
                    cache=False,
                    project=str(project),
                    name=name,
                    exist_ok=True,
                    pretrained=True,
                    plots=True,
                    save=True,
                    verbose=True,
                    val=True,
                )
                print(f"Training returned: {results}")
        duration = time.perf_counter() - started
        best = run_dir / "weights" / "best.pt"
        last = run_dir / "weights" / "last.pt"
        if not best.is_file() or not last.is_file():
            raise FileNotFoundError(
                f"training returned without both checkpoints (best={best.exists()}, last={last.exists()})"
            )
        checkpoint_hash_after = _sha256(base_checkpoint)
        if checkpoint_hash_after != checkpoint_hash_before:
            raise RuntimeError("original pretrained checkpoint changed during training")
        summary.update(
            {
                "status": "completed",
                "duration_seconds": round(duration, 2),
                "best_checkpoint": str(best),
                "best_checkpoint_sha256": _sha256(best),
                "last_checkpoint": str(last),
                "last_checkpoint_sha256": _sha256(last),
                "base_checkpoint_sha256_after": checkpoint_hash_after,
            }
        )
        _json_write(run_dir / "training_summary.json", summary)
        return summary
    except Exception as error:
        duration = time.perf_counter() - started
        summary.update(
            {
                "status": "failed",
                "duration_seconds": round(duration, 2),
                "error_type": type(error).__name__,
                "error": str(error),
                "traceback": traceback.format_exc(),
                "recoverable_best_checkpoint": str(run_dir / "weights" / "best.pt")
                if (run_dir / "weights" / "best.pt").is_file()
                else None,
                "recoverable_last_checkpoint": str(run_dir / "weights" / "last.pt")
                if (run_dir / "weights" / "last.pt").is_file()
                else None,
            }
        )
        _json_write(run_dir / "training_summary.json", summary)
        raise


def _translate_label_line(line: str) -> str:
    fields = line.split()
    if len(fields) != 5:
        raise ValueError(f"expected five YOLO label fields, got {len(fields)}")
    try:
        target_id = int(fields[0])
    except ValueError as error:
        raise ValueError(f"invalid target class id: {fields[0]!r}") from error
    if target_id not in COCO_CLASS_BY_TARGET_ID:
        raise ValueError(f"target class {target_id} has no exact baseline counterpart")
    return " ".join([str(COCO_CLASS_BY_TARGET_ID[target_id]), *fields[1:]])


def _write_baseline_test_dataset(
    *,
    dataset_root: Path,
    baseline_dataset_root: Path,
    baseline_names: dict[int, str],
) -> tuple[Path, dict[int, int]]:
    image_dir = baseline_dataset_root / "images" / "test"
    label_dir = baseline_dataset_root / "labels" / "test"
    image_dir.mkdir(parents=True)
    label_dir.mkdir(parents=True)
    source_images = sorted(
        path for path in (dataset_root / "images" / "test").glob("*")
        if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )
    if not source_images:
        raise ValueError("held-out test split is empty")
    supports: Counter[int] = Counter()
    for image_path in source_images:
        shutil.copy2(image_path, image_dir / image_path.name)
        source_label = dataset_root / "labels" / "test" / f"{image_path.stem}.txt"
        translated = []
        for line in source_label.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            translated_line = _translate_label_line(line)
            translated.append(translated_line)
            supports[int(translated_line.split()[0])] += 1
        (label_dir / f"{image_path.stem}.txt").write_text(
            "\n".join(translated) + ("\n" if translated else ""),
            encoding="utf-8",
        )
    names = {int(key): str(value) for key, value in baseline_names.items()}
    config = {
        "path": str(baseline_dataset_root),
        "train": "images/test",
        "val": "images/test",
        "test": "images/test",
        "names": names,
    }
    config_path = baseline_dataset_root / "data.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return config_path, dict(supports)


def _class_metrics(result: Any, class_names: dict[int, str]) -> dict[int, dict[str, float]]:
    box = result.box
    class_indices = np.asarray(box.ap_class_index, dtype=int).tolist()
    metrics: dict[int, dict[str, float]] = {}
    for metric_index, class_id in enumerate(class_indices):
        if class_id not in class_names:
            continue
        metrics[class_id] = {
            "precision": float(box.p[metric_index]),
            "recall": float(box.r[metric_index]),
            "mAP50": float(box.ap50[metric_index]),
            "mAP50_95": float(box.ap[metric_index]),
        }
    return metrics


def _iou(box_a: np.ndarray, box_b: np.ndarray) -> float:
    x1 = max(float(box_a[0]), float(box_b[0]))
    y1 = max(float(box_a[1]), float(box_b[1]))
    x2 = min(float(box_a[2]), float(box_b[2]))
    y2 = min(float(box_a[3]), float(box_b[3]))
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = max(0.0, float(box_a[2] - box_a[0])) * max(0.0, float(box_a[3] - box_a[1]))
    area_b = max(0.0, float(box_b[2] - box_b[0])) * max(0.0, float(box_b[3] - box_b[1]))
    union = area_a + area_b - intersection
    return intersection / union if union > 0 else 0.0


def _ground_truth_boxes(label_path: Path, width: int, height: int) -> list[tuple[int, np.ndarray]]:
    output = []
    for item in parse_yolo_label(label_path.read_text(encoding="utf-8")):
        x1 = (item["x_center"] - item["width"] / 2) * width
        y1 = (item["y_center"] - item["height"] / 2) * height
        x2 = (item["x_center"] + item["width"] / 2) * width
        y2 = (item["y_center"] + item["height"] / 2) * height
        output.append((item["class_id"], np.array([x1, y1, x2, y2], dtype=np.float32)))
    return output


def _save_failure_previews(
    model: Any,
    *,
    dataset_root: Path,
    preview_dir: Path,
) -> list[dict[str, Any]]:
    image_paths = sorted(
        path for path in (dataset_root / "images" / "test").glob("*")
        if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )
    results = model.predict(
        source=[str(path) for path in image_paths],
        imgsz=IMAGE_SIZE,
        batch=BATCH_SIZE,
        device="cpu",
        conf=0.25,
        iou=VALIDATION_IOU,
        max_det=MAX_DETECTIONS,
        save=False,
        verbose=False,
    )
    cases = []
    for image_path, result in zip(image_paths, results):
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"test image could not be read for preview: {image_path}")
        height, width = image.shape[:2]
        ground_truth = _ground_truth_boxes(
            dataset_root / "labels" / "test" / f"{image_path.stem}.txt",
            width,
            height,
        )
        predictions = []
        if result.boxes is not None:
            xyxy = result.boxes.xyxy.cpu().numpy()
            classes = result.boxes.cls.cpu().numpy().astype(int)
            confidences = result.boxes.conf.cpu().numpy()
            predictions = [
                (int(class_id), np.asarray(box, dtype=np.float32), float(confidence))
                for box, class_id, confidence in zip(xyxy, classes, confidences)
            ]
        matched_ground_truth: set[int] = set()
        matched_predictions: set[int] = set()
        by_class: dict[int, list[int]] = {}
        for index, (class_id, _, confidence) in enumerate(predictions):
            by_class.setdefault(class_id, []).append(index)
        for class_id, prediction_indices in by_class.items():
            prediction_indices.sort(key=lambda index: predictions[index][2], reverse=True)
            gt_indices = [
                index for index, (gt_class, _) in enumerate(ground_truth)
                if gt_class == class_id
            ]
            for prediction_index in prediction_indices:
                if not gt_indices:
                    continue
                best_gt = max(
                    gt_indices,
                    key=lambda index: _iou(predictions[prediction_index][1], ground_truth[index][1]),
                )
                if _iou(predictions[prediction_index][1], ground_truth[best_gt][1]) >= 0.5:
                    matched_predictions.add(prediction_index)
                    matched_ground_truth.add(best_gt)
                    gt_indices.remove(best_gt)
        false_positive_count = len(predictions) - len(matched_predictions)
        false_negative_count = len(ground_truth) - len(matched_ground_truth)
        cases.append(
            {
                "image_id": image_path.stem,
                "image_path": str(image_path),
                "false_positives": false_positive_count,
                "false_negatives": false_negative_count,
                "error_count": false_positive_count + false_negative_count,
                "image": image,
                "ground_truth": ground_truth,
                "predictions": predictions,
                "matched_ground_truth": matched_ground_truth,
                "matched_predictions": matched_predictions,
            }
        )

    preview_dir.mkdir(parents=True)
    selected = [case for case in sorted(cases, key=lambda item: item["error_count"], reverse=True) if case["error_count"]][:8]
    if not selected:
        selected = sorted(cases, key=lambda item: item["image_id"])[:4]
    report = []
    for case in selected:
        image = case["image"].copy()
        height, width = image.shape[:2]
        for index, (class_id, box) in enumerate(case["ground_truth"]):
            p1 = (int(round(box[0])), int(round(box[1])))
            p2 = (int(round(box[2])), int(round(box[3])))
            is_missed = index not in case["matched_ground_truth"]
            color = (0, 255, 255) if is_missed else (0, 190, 0)
            cv2.rectangle(image, p1, p2, color, 2)
            label = f"GT {TARGET_NAMES[class_id]}" + (" FN" if is_missed else "")
            cv2.putText(image, label, (p1[0], max(18, p1[1] - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        for index, (class_id, box, confidence) in enumerate(case["predictions"]):
            p1 = (int(round(box[0])), int(round(box[1])))
            p2 = (int(round(box[2])), int(round(box[3])))
            is_false_positive = index not in case["matched_predictions"]
            color = (0, 0, 255) if is_false_positive else (255, 180, 0)
            cv2.rectangle(image, p1, p2, color, 2)
            label = f"Pred {TARGET_NAMES[class_id]} {confidence:.2f}" + (" FP" if is_false_positive else "")
            cv2.putText(image, label, (p1[0], min(height - 5, p2[1] + 18)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        path = preview_dir / f"{case['image_id']}_errors.jpg"
        if not cv2.imwrite(str(path), image):
            raise OSError(f"could not save annotated failure preview: {path}")
        report.append(
            {
                "image_id": case["image_id"],
                "preview": path.name,
                "false_positives": case["false_positives"],
                "false_negatives": case["false_negatives"],
                "error_count": case["error_count"],
            }
        )
    return report


def _write_evaluation_report(evaluation_dir: Path, result_data: dict[str, Any]) -> None:
    custom = result_data["custom"]
    baseline = result_data["baseline"]
    lines = [
        "# BYAKU YOLO11n Held-Out Test Evaluation",
        "",
        f"- Test set: `{result_data['dataset']}`",
        f"- Test images: {result_data['test_image_count']}",
        f"- Test boxes: {result_data['test_box_count']}",
        f"- Custom checkpoint: `{result_data['training_checkpoint']}`",
        f"- Baseline checkpoint: `{result_data['baseline_checkpoint']}`",
        "- Both models evaluated on the same 60 source test images and the same box geometry.",
        "- Baseline labels were translated to exact COCO classes where available; target `table` is compared to baseline `dining table`.",
        "",
        "## Overall metrics",
        "",
        "| Metric | Custom | Original baseline |",
        "|---|---:|---:|",
        f"| Precision | {custom['precision']:.4f} | {baseline['precision']:.4f} |",
        f"| Recall | {custom['recall']:.4f} | {baseline['recall']:.4f} |",
        f"| mAP@0.5 | {custom['mAP50']:.4f} | {baseline['mAP50']:.4f} |",
        f"| mAP@0.5:0.95 | {custom['mAP50_95']:.4f} | {baseline['mAP50_95']:.4f} |",
        f"| Model-only inference | {custom['inference_ms_per_image']:.2f} ms/image, {custom['fps_model_only']:.2f} FPS | {baseline['inference_ms_per_image']:.2f} ms/image, {baseline['fps_model_only']:.2f} FPS |",
        "",
        "Inference latency uses the same CPU, image size, batch, workers, confidence, IoU, and maximum-detection settings for both models. FPS is model-only and does not represent end-to-end BYAKU real-time throughput.",
        "",
        "## Per-class metrics",
        "",
        "| Class | Test GT support | Custom P | Custom R | Custom mAP@0.5 | Custom mAP@0.5:0.95 | Baseline native class | Baseline P | Baseline R | Baseline mAP@0.5 | Baseline mAP@0.5:0.95 |",
        "|---|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|",
    ]
    for class_name, values in result_data["per_class"].items():
        custom_values = values["custom"] or {}
        baseline_values = values["baseline"] or {}

        def display(metrics: dict[str, Any], key: str) -> str:
            value = metrics.get(key)
            return f"{value:.4f}" if isinstance(value, (int, float)) else "N/A"

        lines.append(
            "| "
            + " | ".join(
                [
                    class_name,
                    str(values["support_boxes"]),
                    display(custom_values, "precision"),
                    display(custom_values, "recall"),
                    display(custom_values, "mAP50"),
                    display(custom_values, "mAP50_95"),
                    values["baseline_native_class"] or "N/A",
                    display(baseline_values, "precision"),
                    display(baseline_values, "recall"),
                    display(baseline_values, "mAP50"),
                    display(baseline_values, "mAP50_95"),
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "`bag` has zero ground-truth boxes in this test set and is not a model output class; no bag detection claim is made.",
            "",
            "## Annotated previews and failures",
            "",
            f"- Ultralytics validation plots: `{evaluation_dir / 'custom_test'}` and `{evaluation_dir / 'baseline_test'}`.",
            f"- Custom predictions with ground truth and FP/FN markings: `{evaluation_dir / 'custom_failure_previews'}`.",
            "- Qualitative preview predictions use confidence 0.25 and IoU 0.7; validation metrics use confidence 0.001.",
            "",
        ]
    )
    (evaluation_dir / "evaluation_report.md").write_text(
        "\n".join(lines),
        encoding="utf-8",
    )


def run_evaluation(
    *,
    custom_checkpoint: Path,
    base_checkpoint: Path = BASE_CHECKPOINT,
    dataset_root: Path = DEFAULT_OUTPUT_ROOT,
    output_root: Path,
    evaluation_name: str | None = None,
) -> dict[str, Any]:
    custom_checkpoint = Path(custom_checkpoint).resolve()
    base_checkpoint = Path(base_checkpoint).resolve()
    dataset_root = Path(dataset_root).resolve()
    output_root = Path(output_root).resolve()
    if not custom_checkpoint.is_file() or not base_checkpoint.is_file():
        raise FileNotFoundError("both custom and original baseline checkpoints must exist")
    if not (dataset_root / "data.yaml").is_file():
        raise FileNotFoundError(f"prepared dataset is missing: {dataset_root}")
    try:
        output_root.relative_to((WORKSPACE_ROOT / "Models").resolve())
    except ValueError as error:
        raise ValueError(f"evaluation output must remain inside {WORKSPACE_ROOT / 'Models'}") from error
    _, split_counts = _validate_dataset(dataset_root / "data.yaml")
    evaluation_name = evaluation_name or f"evaluation_{datetime.now():%Y%m%d_%H%M%S}"
    evaluation_dir = output_root / evaluation_name
    if evaluation_dir.exists():
        raise FileExistsError(f"refusing to overwrite existing evaluation: {evaluation_dir}")
    evaluation_dir.mkdir(parents=True)
    config_dir = WORKSPACE_ROOT / "Config" / "ultralytics"
    config_dir.mkdir(parents=True, exist_ok=True)
    os.environ["YOLO_CONFIG_DIR"] = str(config_dir)
    os.environ["MPLCONFIGDIR"] = str(config_dir / "matplotlib")
    Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
    from ultralytics import YOLO

    custom_model = YOLO(str(custom_checkpoint))
    baseline_model = YOLO(str(base_checkpoint))
    baseline_data, baseline_support = _write_baseline_test_dataset(
        dataset_root=dataset_root,
        baseline_dataset_root=evaluation_dir / "baseline_test_dataset",
        baseline_names=baseline_model.names,
    )
    common_validation = {
        "imgsz": IMAGE_SIZE,
        "batch": BATCH_SIZE,
        "device": "cpu",
        "workers": WORKERS,
        "conf": VALIDATION_CONFIDENCE,
        "iou": VALIDATION_IOU,
        "max_det": MAX_DETECTIONS,
        "plots": True,
        "save_json": True,
        "verbose": True,
    }
    custom_result = custom_model.val(
        data=str(dataset_root / "data.yaml"),
        split="test",
        project=str(evaluation_dir),
        name="custom_test",
        exist_ok=False,
        **common_validation,
    )
    baseline_result = baseline_model.val(
        data=str(baseline_data),
        split="test",
        project=str(evaluation_dir),
        name="baseline_test",
        exist_ok=False,
        **common_validation,
    )

    custom_class_metrics = _class_metrics(custom_result, TARGET_NAMES)
    baseline_source_names = {class_id: str(name) for class_id, name in baseline_model.names.items()}
    baseline_class_metrics_by_source = _class_metrics(baseline_result, baseline_source_names)
    baseline_class_metrics = {
        target_id: baseline_class_metrics_by_source.get(source_id)
        for target_id, source_id in COCO_CLASS_BY_TARGET_ID.items()
    }
    support_by_target = {
        class_id: int(split_counts[f"test_{TARGET_NAMES[class_id]}"])
        for class_id in TARGET_NAMES
    }
    support_by_target[7] = 0
    per_class = {}
    for target_id in range(8):
        target_name = (
            TARGET_NAMES[target_id]
            if target_id in TARGET_NAMES
            else "bag"
        )
        per_class[target_name] = {
            "support_boxes": support_by_target[target_id],
            "custom": custom_class_metrics.get(target_id) if support_by_target[target_id] else None,
            "baseline": baseline_class_metrics.get(target_id) if target_id in baseline_class_metrics else None,
            "baseline_native_class": (
                baseline_source_names.get(COCO_CLASS_BY_TARGET_ID[target_id])
                if target_id in COCO_CLASS_BY_TARGET_ID
                else None
            ),
            "status": (
                "unsupported_no_test_ground_truth"
                if support_by_target[target_id] == 0
                else "evaluated"
            ),
        }
    custom_metrics = {
        "precision": float(custom_result.results_dict["metrics/precision(B)"]),
        "recall": float(custom_result.results_dict["metrics/recall(B)"]),
        "mAP50": float(custom_result.results_dict["metrics/mAP50(B)"]),
        "mAP50_95": float(custom_result.results_dict["metrics/mAP50-95(B)"]),
        "inference_ms_per_image": float(custom_result.speed["inference"]),
    }
    baseline_metrics = {
        "precision": float(baseline_result.results_dict["metrics/precision(B)"]),
        "recall": float(baseline_result.results_dict["metrics/recall(B)"]),
        "mAP50": float(baseline_result.results_dict["metrics/mAP50(B)"]),
        "mAP50_95": float(baseline_result.results_dict["metrics/mAP50-95(B)"]),
        "inference_ms_per_image": float(baseline_result.speed["inference"]),
    }
    for metric in (custom_metrics, baseline_metrics):
        latency = metric["inference_ms_per_image"]
        metric["fps_model_only"] = 1000.0 / latency if latency > 0 else None
    failure_previews = _save_failure_previews(
        custom_model,
        dataset_root=dataset_root,
        preview_dir=evaluation_dir / "custom_failure_previews",
    )
    result_data = {
        "dataset": str(dataset_root),
        "test_image_count": split_counts["test_images"],
        "test_box_count": sum(support_by_target.values()),
        "test_class_support": {
            name: support_by_target[class_id]
            for class_id, name in {**TARGET_NAMES, 7: "bag"}.items()
        },
        "training_checkpoint": str(custom_checkpoint),
        "baseline_checkpoint": str(base_checkpoint),
        "baseline_class_map": {
            TARGET_NAMES[target_id]: {
                "target_id": target_id,
                "baseline_id": source_id,
                "baseline_native_name": baseline_source_names[source_id],
            }
            for target_id, source_id in COCO_CLASS_BY_TARGET_ID.items()
        },
        "baseline_table_limitation": (
            "Open Images target 'table' maps to COCO 'dining table'; the baseline has no exact generic table class."
        ),
        "settings": {
            "imgsz": IMAGE_SIZE,
            "batch": BATCH_SIZE,
            "device": "cpu",
            "workers": WORKERS,
            "confidence": VALIDATION_CONFIDENCE,
            "iou": VALIDATION_IOU,
            "max_det": MAX_DETECTIONS,
            "inference_latency_scope": "model-only Ultralytics validation inference speed; excludes BYAKU pipeline overhead",
        },
        "custom": custom_metrics,
        "baseline": baseline_metrics,
        "per_class": per_class,
        "baseline_test_support_by_source_id": baseline_support,
        "custom_failure_previews": failure_previews,
    }
    _json_write(evaluation_dir / "evaluation_metrics.json", result_data)
    with (evaluation_dir / "per_class_metrics.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=[
                "class",
                "support_boxes",
                "status",
                "custom_precision",
                "custom_recall",
                "custom_mAP50",
                "custom_mAP50_95",
                "baseline_native_class",
                "baseline_precision",
                "baseline_recall",
                "baseline_mAP50",
                "baseline_mAP50_95",
            ],
        )
        writer.writeheader()
        for class_name, row in per_class.items():
            custom = row["custom"] or {}
            baseline = row["baseline"] or {}
            writer.writerow(
                {
                    "class": class_name,
                    "support_boxes": row["support_boxes"],
                    "status": row["status"],
                    "custom_precision": custom.get("precision"),
                    "custom_recall": custom.get("recall"),
                    "custom_mAP50": custom.get("mAP50"),
                    "custom_mAP50_95": custom.get("mAP50_95"),
                    "baseline_native_class": row["baseline_native_class"],
                    "baseline_precision": baseline.get("precision"),
                    "baseline_recall": baseline.get("recall"),
                    "baseline_mAP50": baseline.get("mAP50"),
                    "baseline_mAP50_95": baseline.get("mAP50_95"),
                }
            )
    _write_evaluation_report(evaluation_dir, result_data)
    return result_data


def main() -> int:
    parser = argparse.ArgumentParser(description="Train or evaluate a BYAKU-only YOLO11n model.")
    subparsers = parser.add_subparsers(dest="command", required=True)
    train_parser = subparsers.add_parser("train")
    train_parser.add_argument("--name")
    train_parser.add_argument("--data-yaml", type=Path, default=DEFAULT_OUTPUT_ROOT / "data.yaml")
    train_parser.add_argument("--base-checkpoint", type=Path, default=BASE_CHECKPOINT)
    train_parser.add_argument("--project", type=Path, default=DEFAULT_PROJECT)
    eval_parser = subparsers.add_parser("evaluate")
    eval_parser.add_argument("--custom-checkpoint", type=Path, required=True)
    eval_parser.add_argument("--base-checkpoint", type=Path, default=BASE_CHECKPOINT)
    eval_parser.add_argument("--dataset-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    eval_parser.add_argument("--output-root", type=Path, required=True)
    eval_parser.add_argument("--evaluation-name")
    args = parser.parse_args()
    if args.command == "train":
        result = run_training(
            name=args.name,
            data_yaml=args.data_yaml,
            base_checkpoint=args.base_checkpoint,
            project=args.project,
        )
    else:
        result = run_evaluation(
            custom_checkpoint=args.custom_checkpoint,
            base_checkpoint=args.base_checkpoint,
            dataset_root=args.dataset_root,
            output_root=args.output_root,
            evaluation_name=args.evaluation_name,
        )
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
