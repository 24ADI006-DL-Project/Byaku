import json
from pathlib import Path

import cv2
import numpy as np
import pytest
import yaml

from Workflow.tools.dataset_validation import (
    CLASS_NAMES,
    SPLITS,
    _write_data_yaml,
    deterministic_group_split,
    parse_yolo_label,
    sha256_file,
    validate_dataset,
)


def _create_dataset(root: Path) -> Path:
    for split in SPLITS:
        (root / "images" / split).mkdir(parents=True)
        (root / "labels" / split).mkdir(parents=True)
    (root / "metadata").mkdir()
    _write_data_yaml(root)
    return root


def _write_split_manifest(root: Path) -> None:
    split_paths = {}
    entries = []
    for split in SPLITS:
        paths = sorted((root / "images" / split).glob("*"))
        rel_paths = [path.relative_to(root).as_posix() for path in paths if path.is_file()]
        split_paths[split] = rel_paths
        entries.extend(
            {"image_path": image_path, "source_group": image_path}
            for image_path in rel_paths
        )
    (root / "metadata" / "split_manifest.json").write_text(
        json.dumps({"splits": split_paths, "images": entries}),
        encoding="utf-8",
    )


def _write_image(path: Path, value: int = 20) -> None:
    image = np.full((128, 160, 3), value, dtype=np.uint8)
    assert cv2.imwrite(str(path), image)


def test_target_class_vocabulary_and_yaml_are_exact(tmp_path):
    dataset = _create_dataset(tmp_path / "dataset")
    loaded = yaml.safe_load((dataset / "data.yaml").read_text(encoding="utf-8"))

    assert CLASS_NAMES == (
        "person", "bottle", "book", "chair", "table", "mobile_phone", "laptop", "bag"
    )
    assert loaded["names"] == dict(enumerate(CLASS_NAMES))
    assert loaded["train"] == "images/train"
    assert loaded["val"] == "images/val"
    assert loaded["test"] == "images/test"


def test_yolo_label_parser_accepts_valid_normalized_box():
    assert parse_yolo_label("5 0.5 0.5 0.4 0.6\n") == [
        {
            "class_id": 5,
            "x_center": 0.5,
            "y_center": 0.5,
            "width": 0.4,
            "height": 0.6,
            "line": 1,
        }
    ]


@pytest.mark.parametrize(
    ("label", "message"),
    [
        ("8 0.5 0.5 0.2 0.2", "class_id 8"),
        ("1 1.1 0.5 0.2 0.2", "center coordinates"),
        ("1 0.5 0.5 0 0.2", "width and height"),
        ("1 0.1 0.5 0.4 0.2", "outside image bounds"),
        ("1 0.5 nan 0.2 0.2", "finite"),
        ("1 0.5 0.5 0.2", "expected 5 fields"),
    ],
)
def test_yolo_label_parser_rejects_invalid_classes_and_coordinates(label, message):
    with pytest.raises(ValueError, match=message):
        parse_yolo_label(label)


def test_grouped_split_is_deterministic_and_keeps_video_sequence_together():
    items = [
        {"image_path": f"frame-{index:02}.jpg", "source_group": "video-a"}
        for index in range(20)
    ] + [
        {"image_path": "still-1.jpg", "source_group": "still-1"},
        {"image_path": "still-2.jpg", "source_group": "still-2"},
    ]

    first = deterministic_group_split(items, seed=39)
    second = deterministic_group_split(items, seed=39)

    assert first == second
    assert first["counts"] == {"train": 20, "val": 1, "test": 1}
    assert first["source_groups"]["video-a"] in SPLITS
    assert sum(
        len(paths)
        for paths in first["images"].values()
    ) == len(items)
    for split, paths in first["images"].items():
        if any(path.startswith("frame-") for path in paths):
            assert all(path.startswith("frame-") for path in paths)


def test_sha256_hash_detects_exact_duplicate_files(tmp_path):
    first = tmp_path / "first.jpg"
    second = tmp_path / "second.jpg"
    _write_image(first, 10)
    second.write_bytes(first.read_bytes())

    assert sha256_file(first) == sha256_file(second)


def test_dataset_validation_accepts_paired_images_and_labels(tmp_path):
    dataset = _create_dataset(tmp_path / "dataset")
    image_path = dataset / "images" / "train" / "person.jpg"
    label_path = dataset / "labels" / "train" / "person.txt"
    _write_image(image_path)
    label_path.write_text("0 0.5 0.5 0.4 0.6\n", encoding="utf-8")
    _write_split_manifest(dataset)

    result = validate_dataset(dataset)

    assert result["quality_report"]["valid"] is True
    assert result["quality_report"]["annotation_counts"] == {
        "annotated_image_count": 1,
        "bounding_box_count": 1,
    }
    assert result["quality_report"]["class_distribution"]["person"]["boxes"] == 1
    assert result["quality_report"]["image_counts_by_split"] == {
        "train": 1, "val": 0, "test": 0
    }


def test_dataset_validation_reports_missing_label_and_returns_invalid(tmp_path):
    dataset = _create_dataset(tmp_path / "dataset")
    _write_image(dataset / "images" / "train" / "unlabeled.jpg")
    _write_split_manifest(dataset)

    result = validate_dataset(dataset)

    assert result["quality_report"]["valid"] is False
    assert result["quality_report"]["missing_labels"] == ["images/train/unlabeled.jpg"]
    assert "one or more images are missing labels" in result["quality_report"]["structural_errors"]


def test_dataset_validation_reports_invalid_class_and_box_bounds(tmp_path):
    dataset = _create_dataset(tmp_path / "dataset")
    image_path = dataset / "images" / "train" / "bad.jpg"
    _write_image(image_path)
    (dataset / "labels" / "train" / "bad.txt").write_text(
        "8 0.5 0.5 0.2 0.2\n", encoding="utf-8"
    )
    _write_split_manifest(dataset)

    result = validate_dataset(dataset)

    assert result["quality_report"]["valid"] is False
    assert result["quality_report"]["malformed_labels"][0]["reason"].startswith(
        "line 1: class_id 8"
    )


def test_dataset_validation_detects_exact_duplicates_and_orphan_labels(tmp_path):
    dataset = _create_dataset(tmp_path / "dataset")
    first = dataset / "images" / "train" / "first.jpg"
    second = dataset / "images" / "train" / "second.jpg"
    _write_image(first, 10)
    second.write_bytes(first.read_bytes())
    for name in ("first", "second"):
        (dataset / "labels" / "train" / f"{name}.txt").write_text(
            "0 0.5 0.5 0.2 0.2\n", encoding="utf-8"
        )
    (dataset / "labels" / "train" / "orphan.txt").write_text(
        "0 0.5 0.5 0.2 0.2\n", encoding="utf-8"
    )
    _write_split_manifest(dataset)

    report = validate_dataset(dataset)["quality_report"]

    assert report["duplicate_images"]["exact_duplicate_count"] == 1
    assert report["orphan_labels"] == ["labels/train/orphan.txt"]
    assert report["valid"] is False


def test_dataset_validation_rejects_source_group_leakage(tmp_path):
    dataset = _create_dataset(tmp_path / "dataset")
    entries = []
    split_paths = {split: [] for split in SPLITS}
    for split, filename in (("train", "frame-a.jpg"), ("val", "frame-b.jpg")):
        image_path = dataset / "images" / split / filename
        _write_image(image_path)
        (dataset / "labels" / split / f"{Path(filename).stem}.txt").write_text(
            "0 0.5 0.5 0.2 0.2\n", encoding="utf-8"
        )
        relative = image_path.relative_to(dataset).as_posix()
        split_paths[split].append(relative)
        entries.append({"image_path": relative, "source_group": "same-video"})
    (dataset / "metadata" / "split_manifest.json").write_text(
        json.dumps({"splits": split_paths, "images": entries}),
        encoding="utf-8",
    )

    report = validate_dataset(dataset)["quality_report"]

    assert report["source_group_split_leaks"] == [
        {"source_group": "same-video", "splits": ["train", "val"]}
    ]
    assert report["valid"] is False
