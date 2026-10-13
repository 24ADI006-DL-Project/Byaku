from types import SimpleNamespace

import cv2
import numpy as np
import pytest
import yaml

from Workflow.training.prepare_open_images_training import TRAINABLE_CLASSES
from Workflow.training.run_detector_training import (
    TARGET_NAMES,
    _class_metrics,
    _translate_label_line,
    _validate_dataset,
    _write_baseline_test_dataset,
    _write_evaluation_report,
)


def test_baseline_label_translation_uses_only_exact_class_map():
    assert _translate_label_line("4 0.5 0.5 0.2 0.3") == "60 0.5 0.5 0.2 0.3"
    assert _translate_label_line("5 0.5 0.5 0.2 0.3") == "67 0.5 0.5 0.2 0.3"
    with pytest.raises(ValueError, match="no exact baseline counterpart"):
        _translate_label_line("7 0.5 0.5 0.2 0.3")


def test_baseline_test_data_uses_native_coco_class_ids(tmp_path):
    dataset_root = tmp_path / "prepared"
    image_dir = dataset_root / "images" / "test"
    label_dir = dataset_root / "labels" / "test"
    image_dir.mkdir(parents=True)
    label_dir.mkdir(parents=True)
    assert cv2.imwrite(str(image_dir / "frame.jpg"), np.full((24, 32, 3), 100, dtype=np.uint8))
    (label_dir / "frame.txt").write_text(
        "4 0.5 0.5 0.2 0.2\n5 0.5 0.5 0.2 0.2\n",
        encoding="utf-8",
    )
    names = {index: f"class-{index}" for index in range(80)}

    data_yaml, support = _write_baseline_test_dataset(
        dataset_root=dataset_root,
        baseline_dataset_root=tmp_path / "baseline",
        baseline_names=names,
    )

    baseline_labels = (tmp_path / "baseline" / "labels" / "test" / "frame.txt").read_text()
    baseline_config = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))
    assert baseline_labels.splitlines() == [
        "60 0.5 0.5 0.2 0.2",
        "67 0.5 0.5 0.2 0.2",
    ]
    assert support == {60: 1, 67: 1}
    assert len(baseline_config["names"]) == 80
    assert baseline_config["test"] == baseline_config["val"] == "images/test"


def test_metrics_use_actual_sparse_baseline_class_ids():
    result = SimpleNamespace(
        box=SimpleNamespace(
            ap_class_index=np.array([39, 73]),
            p=np.array([0.2, 0.3]),
            r=np.array([0.4, 0.5]),
            ap50=np.array([0.6, 0.7]),
            ap=np.array([0.1, 0.2]),
        )
    )

    metrics = _class_metrics(result, {39: "bottle", 73: "book"})

    assert metrics[39] == {
        "precision": 0.2,
        "recall": 0.4,
        "mAP50": 0.6,
        "mAP50_95": 0.1,
    }
    assert metrics[73]["mAP50"] == 0.7


def test_evaluation_report_marks_bag_unsupported_and_fps_model_only(tmp_path):
    metrics = {
        "precision": 0.5,
        "recall": 0.4,
        "mAP50": 0.3,
        "mAP50_95": 0.2,
        "inference_ms_per_image": 50.0,
        "fps_model_only": 20.0,
    }
    result_data = {
        "dataset": "prepared dataset",
        "test_image_count": 60,
        "test_box_count": 10,
        "training_checkpoint": "best.pt",
        "baseline_checkpoint": "Models/yolo11n.pt",
        "custom": metrics,
        "baseline": metrics,
        "per_class": {
            "person": {
                "support_boxes": 10,
                "custom": metrics,
                "baseline": metrics,
                "baseline_native_class": "person",
            },
            "bag": {
                "support_boxes": 0,
                "custom": None,
                "baseline": None,
                "baseline_native_class": None,
            },
        },
    }

    _write_evaluation_report(tmp_path, result_data)

    report = (tmp_path / "evaluation_report.md").read_text(encoding="utf-8")
    assert "`bag` has zero ground-truth boxes" in report
    assert "model-only" in report
    assert "| bag | 0 | N/A | N/A |" in report


def test_training_dataset_validation_checks_all_splits_and_class_support(tmp_path):
    names = {item["target_id"]: item["name"] for item in TRAINABLE_CLASSES}
    for split_index, split in enumerate(("train", "val", "test")):
        image_dir = tmp_path / "images" / split
        label_dir = tmp_path / "labels" / split
        image_dir.mkdir(parents=True)
        label_dir.mkdir(parents=True)
        image = np.full((32, 48, 3), 100 + split_index * 40, dtype=np.uint8)
        assert cv2.imwrite(str(image_dir / "sample.jpg"), image)
        label_lines = [
            f"{class_id} 0.5 0.5 0.2 0.2"
            for class_id in range(7)
        ] if split == "train" else ["0 0.5 0.5 0.2 0.2"]
        (label_dir / "sample.txt").write_text("\n".join(label_lines) + "\n", encoding="utf-8")
    data_yaml = tmp_path / "data.yaml"
    data_yaml.write_text(
        yaml.safe_dump(
            {
                "path": str(tmp_path),
                "train": "images/train",
                "val": "images/val",
                "test": "images/test",
                "names": names,
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    config, counts = _validate_dataset(data_yaml)

    assert config["test"] == "images/test"
    assert counts["train_images"] == counts["val_images"] == counts["test_images"] == 1
    assert all(counts[f"train_{name}"] == 1 for name in TARGET_NAMES.values())


def test_training_dataset_validation_rejects_test_train_image_leakage(tmp_path):
    names = {item["target_id"]: item["name"] for item in TRAINABLE_CLASSES}
    source = tmp_path / "same.jpg"
    assert cv2.imwrite(str(source), np.full((32, 48, 3), 100, dtype=np.uint8))
    for split in ("train", "val", "test"):
        image_dir = tmp_path / "images" / split
        label_dir = tmp_path / "labels" / split
        image_dir.mkdir(parents=True)
        label_dir.mkdir(parents=True)
        (image_dir / "sample.jpg").write_bytes(source.read_bytes())
        label_lines = [
            f"{class_id} 0.5 0.5 0.2 0.2"
            for class_id in range(7)
        ] if split == "train" else ["0 0.5 0.5 0.2 0.2"]
        (label_dir / "sample.txt").write_text("\n".join(label_lines) + "\n", encoding="utf-8")
    data_yaml = tmp_path / "data.yaml"
    data_yaml.write_text(
        yaml.safe_dump(
            {
                "path": str(tmp_path),
                "train": "images/train",
                "val": "images/val",
                "test": "images/test",
                "names": names,
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="duplicate image leakage"):
        _validate_dataset(data_yaml)
