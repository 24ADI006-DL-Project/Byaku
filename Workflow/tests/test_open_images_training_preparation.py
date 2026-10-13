import json
from pathlib import Path

import numpy as np
from PIL import Image
import pytest
import yaml

from Workflow.training.prepare_open_images_training import (
    TARGET_CLASSES,
    prepare_dataset,
)


def _write_source_tree(root: Path, *, duplicate: bool = False) -> Path:
    image_patterns = {
        "train": np.tile(np.arange(96, dtype=np.uint8), (64, 1)),
        "val": np.tile(np.arange(64, dtype=np.uint8)[:, None], (1, 96)),
        "test": np.indices((64, 96)).sum(axis=0).astype(np.uint8) * 4,
    }
    manifest = []
    for index, split in enumerate(("train", "val", "test")):
        image_id = f"image-{split}"
        image_dir = root / "byaku_yolo_v7" / "images" / split
        label_dir = root / "byaku_yolo_v7" / "labels" / split
        image_dir.mkdir(parents=True, exist_ok=True)
        label_dir.mkdir(parents=True, exist_ok=True)
        image_path = image_dir / f"{image_id}.jpg"
        pixels = image_patterns[split]
        if duplicate and split == "val":
            pixels = image_patterns["train"]
        Image.fromarray(pixels).save(image_path, quality=95)
        boxes = [
            {
                "class_id": item["target_id"],
                "class_mid": item["mid"],
                "class_name": item["name"],
                "xmin": 0.1,
                "xmax": 0.9,
                "ymin": 0.1,
                "ymax": 0.9,
            }
            for item in TARGET_CLASSES[:7]
        ] if index == 0 else [
            {
                "class_id": 0,
                "class_mid": TARGET_CLASSES[0]["mid"],
                "class_name": TARGET_CLASSES[0]["name"],
                "xmin": 0.2,
                "xmax": 0.8,
                "ymin": 0.2,
                "ymax": 0.8,
            }
        ]
        lines = [
            f"{box['class_id']} 0.500000 0.500000 0.800000 0.800000"
            if index == 0
            else f"0 0.500000 0.500000 0.600000 0.600000"
            for box in boxes
        ]
        label_path = label_dir / f"{image_id}.txt"
        label_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        manifest.append(
            {
                "image_id": image_id,
                "split": split,
                "image_path": str(Path("byaku_yolo_v7") / "images" / split / image_path.name),
                "label_path": str(Path("byaku_yolo_v7") / "labels" / split / label_path.name),
                "width": 96,
                "height": 64,
                "license": "https://creativecommons.org/licenses/by/2.0/",
                "author": "Dataset Author",
                "author_profile": "https://example.invalid/author",
                "title": image_id,
                "original_landing_url": f"https://example.invalid/images/{image_id}",
                "boxes": boxes,
            }
        )
    manifest_dir = root / "manifests"
    manifest_dir.mkdir(parents=True)
    manifest_path = manifest_dir / "byaku_full_dataset_manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    return manifest_path


def test_preparation_copies_only_supported_classes_and_keeps_attribution(tmp_path):
    source = tmp_path / "source"
    manifest_path = _write_source_tree(source)
    output = tmp_path / "prepared"

    report = prepare_dataset(
        source,
        output,
        workspace_root=tmp_path,
        manifest_path=manifest_path,
    )

    config = yaml.safe_load((output / "data.yaml").read_text(encoding="utf-8"))
    mapping = json.loads((output / "metadata" / "class_mapping.json").read_text(encoding="utf-8"))
    assert report["image_counts"] == {"train": 1, "val": 1, "test": 1}
    assert report["valid_box_count"] == 9
    assert report["class_box_counts"]["bag"] == 0
    assert config["names"] == {i: entry["name"] for i, entry in enumerate(TARGET_CLASSES[:7])}
    assert mapping["target_classes"][7]["target_id"] == 7
    assert mapping["target_classes"][7]["training_id"] is None
    assert mapping["target_classes"][7]["trainable"] is False
    assert len(json.loads((output / "metadata" / "source_attributions.json").read_text())) == 3
    assert len(list((output / "images" / "train").glob("*.jpg"))) == 1


def test_preparation_rejects_exact_duplicates_before_creating_output(tmp_path):
    source = tmp_path / "source"
    manifest_path = _write_source_tree(source, duplicate=True)
    output = tmp_path / "prepared"

    with pytest.raises(ValueError, match="duplicate image content"):
        prepare_dataset(source, output, workspace_root=tmp_path, manifest_path=manifest_path)

    assert not output.exists()


def test_preparation_refuses_overwrite_and_output_outside_workspace(tmp_path):
    source = tmp_path / "source"
    manifest_path = _write_source_tree(source)
    output = tmp_path / "prepared"
    output.mkdir()
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        prepare_dataset(source, output, workspace_root=tmp_path, manifest_path=manifest_path)

    with pytest.raises(ValueError, match="inside"):
        prepare_dataset(
            source,
            tmp_path.parent / "outside",
            workspace_root=tmp_path,
            manifest_path=manifest_path,
        )
