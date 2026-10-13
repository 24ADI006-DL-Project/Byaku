from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any

import numpy as np
from PIL import Image
import yaml

from Workflow.tools.dataset_validation import SPLITS, parse_yolo_label


WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SOURCE_ROOT = WORKSPACE_ROOT / "Data" / "datasets" / "open_images_v7"
DEFAULT_OUTPUT_ROOT = WORKSPACE_ROOT / "Data" / "datasets" / "prepared_open_images_v7_full600"
MANIFEST_NAME = "byaku_full_dataset_manifest.json"
TARGET_CLASSES = (
    {"target_id": 0, "name": "person", "mid": "/m/01g317"},
    {"target_id": 1, "name": "bottle", "mid": "/m/04dr76w"},
    {"target_id": 2, "name": "book", "mid": "/m/0bt_c3"},
    {"target_id": 3, "name": "chair", "mid": "/m/01mzpv"},
    {"target_id": 4, "name": "table", "mid": "/m/04bcr3"},
    {"target_id": 5, "name": "mobile_phone", "mid": "/m/050k8"},
    {"target_id": 6, "name": "laptop", "mid": "/m/01c648"},
    {"target_id": 7, "name": "bag", "mid": "/m/0n5v01m"},
)
TRAINABLE_CLASSES = TARGET_CLASSES[:7]
_DCT_SIZE = 32
_DCT_FREQUENCIES = 8
_frequencies = np.arange(_DCT_SIZE)[:, None]
_samples = np.arange(_DCT_SIZE)[None, :]
_DCT_MATRIX = np.cos(
    (2 * _samples + 1) * _frequencies * math.pi / (2 * _DCT_SIZE)
)
_DCT_MATRIX[0] *= 1 / math.sqrt(2)
_DCT_MATRIX *= math.sqrt(2 / _DCT_SIZE)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _perceptual_hash(path: Path) -> int:
    with Image.open(path) as image:
        pixels = np.asarray(
            image.convert("L").resize((_DCT_SIZE, _DCT_SIZE)),
            dtype=np.float32,
        )
    coefficients = (_DCT_MATRIX @ pixels @ _DCT_MATRIX.T)[:_DCT_FREQUENCIES, :_DCT_FREQUENCIES]
    flattened = coefficients.flatten()
    median = float(np.median(flattened[1:]))
    result = 0
    for bit, value in enumerate(flattened):
        if value > median:
            result |= 1 << bit
    return result


def _source_path(source_root: Path, manifest_path: str) -> Path:
    return source_root / manifest_path.replace("\\", os.sep).replace("/", os.sep)


def _validate_source_label(
    label_path: Path,
    boxes: list[dict[str, Any]],
    image_id: str,
) -> None:
    lines = label_path.read_text(encoding="utf-8").splitlines()
    if len(lines) != len(boxes):
        raise ValueError(f"{image_id}: source label line count does not match the manifest")
    for line_number, (line, box) in enumerate(zip(lines, boxes), start=1):
        fields = line.split()
        if len(fields) != 5:
            raise ValueError(f"{image_id}: source label line {line_number} must have 5 fields")
        try:
            class_id = int(fields[0])
            values = [float(value) for value in fields[1:]]
        except ValueError as error:
            raise ValueError(f"{image_id}: source label line {line_number} is not numeric") from error
        expected = (
            (box["xmin"] + box["xmax"]) / 2,
            (box["ymin"] + box["ymax"]) / 2,
            box["xmax"] - box["xmin"],
            box["ymax"] - box["ymin"],
        )
        if class_id != box.get("class_id") or any(
            not math.isclose(actual, wanted, abs_tol=2e-5, rel_tol=0)
            for actual, wanted in zip(values, expected)
        ):
            raise ValueError(f"{image_id}: source label line {line_number} differs from the manifest")


def _box_to_yolo(box: dict[str, Any]) -> str:
    class_id = box["class_id"]
    if class_id not in range(len(TRAINABLE_CLASSES)):
        raise ValueError(f"source box class_id {class_id!r} has no eligible training boxes")
    xmin, xmax = float(box["xmin"]), float(box["xmax"])
    ymin, ymax = float(box["ymin"]), float(box["ymax"])
    if (
        not all(math.isfinite(value) and 0 <= value <= 1 for value in (xmin, xmax, ymin, ymax))
        or xmax <= xmin
        or ymax <= ymin
    ):
        raise ValueError(f"source box for class {class_id} is outside valid image bounds")
    x_center = (xmin + xmax) / 2
    y_center = (ymin + ymax) / 2
    width = xmax - xmin
    height = ymax - ymin
    return f"{class_id} {x_center:.12f} {y_center:.12f} {width:.12f} {height:.12f}"


def prepare_dataset(
    source_root: Path = DEFAULT_SOURCE_ROOT,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    *,
    workspace_root: Path = WORKSPACE_ROOT,
    manifest_path: Path | None = None,
) -> dict[str, Any]:
    """Create an isolated, validated seven-class dataset from the reviewed 600-image manifest."""
    source_root = Path(source_root).resolve()
    output_root = Path(output_root).resolve()
    workspace_root = Path(workspace_root).resolve()
    try:
        output_root.relative_to(workspace_root)
    except ValueError as error:
        raise ValueError(f"output dataset must remain inside {workspace_root}") from error
    if output_root.exists():
        raise FileExistsError(f"refusing to overwrite existing dataset: {output_root}")

    source_manifest = Path(manifest_path) if manifest_path else source_root / "manifests" / MANIFEST_NAME
    records = json.loads(source_manifest.read_text(encoding="utf-8"))
    if not isinstance(records, list) or not records:
        raise ValueError("source manifest must be a non-empty list")

    target_by_id = {entry["target_id"]: entry for entry in TARGET_CLASSES}
    target_by_mid = {entry["mid"]: entry for entry in TARGET_CLASSES}
    seen_ids: set[str] = set()
    exact_hashes: dict[str, str] = {}
    hashes_by_split: list[tuple[str, str, int]] = []
    class_box_counts: Counter[int] = Counter()
    class_image_counts: dict[int, set[str]] = defaultdict(set)
    split_counts: Counter[str] = Counter()
    split_boxes: dict[str, Counter[int]] = defaultdict(Counter)
    prepared_records: list[dict[str, Any]] = []

    for record in records:
        image_id = record.get("image_id")
        split = record.get("split")
        if not isinstance(image_id, str) or not image_id or image_id in seen_ids:
            raise ValueError(f"missing or duplicate source image_id: {image_id!r}")
        seen_ids.add(image_id)
        if split not in SPLITS:
            raise ValueError(f"{image_id}: unsupported split {split!r}")
        if record.get("license") != "https://creativecommons.org/licenses/by/2.0/":
            raise ValueError(f"{image_id}: image-level CC BY 2.0 evidence is missing")
        if not all(
            isinstance(record.get(field), str) and record[field].strip()
            for field in ("author", "title", "original_landing_url")
        ):
            raise ValueError(f"{image_id}: required attribution metadata is missing")

        source_image = _source_path(source_root, record["image_path"])
        source_label = _source_path(source_root, record["label_path"])
        if not source_image.is_file() or not source_label.is_file():
            raise FileNotFoundError(f"{image_id}: source image or label is missing")
        with Image.open(source_image) as image:
            image.verify()
        with Image.open(source_image) as image:
            if image.size != (record.get("width"), record.get("height")):
                raise ValueError(f"{image_id}: image dimensions differ from the manifest")

        boxes = record.get("boxes")
        if not isinstance(boxes, list):
            raise ValueError(f"{image_id}: manifest boxes must be a list")
        for box in boxes:
            class_id = box.get("class_id")
            expected_class = target_by_id.get(class_id)
            if (
                expected_class is None
                or box.get("class_mid") != expected_class["mid"]
                or box.get("class_name") != expected_class["name"]
            ):
                raise ValueError(f"{image_id}: class ID/name/MID mapping is inconsistent")
            if class_id == 7:
                raise ValueError("bag has no valid source boxes and cannot be trained")
            class_box_counts[class_id] += 1
            split_boxes[split][class_id] += 1
            class_image_counts[class_id].add(image_id)
        _validate_source_label(source_label, boxes, image_id)

        digest = _sha256(source_image)
        if digest in exact_hashes:
            raise ValueError(
                f"duplicate image content: {image_id} and {exact_hashes[digest]}"
            )
        exact_hashes[digest] = image_id
        hashes_by_split.append((split, image_id, _perceptual_hash(source_image)))
        split_counts[split] += 1
        prepared_records.append(
            {
                "image_id": image_id,
                "split": split,
                "source_image": source_image,
                "source_image_relpath": record["image_path"],
                "boxes": boxes,
                "license": record["license"],
                "author": record["author"],
                "author_profile": record.get("author_profile"),
                "title": record["title"],
                "landing_url": record["original_landing_url"],
                "sha256": digest,
            }
        )

    if set(split_counts) != set(SPLITS):
        raise ValueError("all train, val, and test splits must be present")
    if any(class_box_counts[class_id] == 0 for class_id in range(7)):
        raise ValueError("one or more trainable target classes have no valid boxes")
    near_duplicate_pairs = []
    for index, (split_a, image_a, hash_a) in enumerate(hashes_by_split):
        for split_b, image_b, hash_b in hashes_by_split[index + 1 :]:
            distance = (hash_a ^ hash_b).bit_count()
            if split_a != split_b and distance <= 4:
                near_duplicate_pairs.append(
                    {
                        "image_a": image_a,
                        "split_a": split_a,
                        "image_b": image_b,
                        "split_b": split_b,
                        "phash_hamming_distance": distance,
                    }
                )
    if near_duplicate_pairs:
        raise ValueError(f"cross-split near-duplicates detected: {near_duplicate_pairs[:5]}")

    output_root.parent.mkdir(parents=True, exist_ok=True)
    staging_root = Path(
        tempfile.mkdtemp(prefix=f".{output_root.name}.staging-", dir=output_root.parent)
    )
    try:
        for split in SPLITS:
            (staging_root / "images" / split).mkdir(parents=True)
            (staging_root / "labels" / split).mkdir(parents=True)
        manifest_entries = []
        attribution_entries = []
        for record in prepared_records:
            image_id = record["image_id"]
            split = record["split"]
            image_rel = Path("images") / split / f"{image_id}.jpg"
            label_rel = Path("labels") / split / f"{image_id}.txt"
            shutil.copy2(record["source_image"], staging_root / image_rel)
            label_text = "\n".join(_box_to_yolo(box) for box in record["boxes"])
            if label_text:
                label_text += "\n"
            (staging_root / label_rel).write_text(label_text, encoding="utf-8")
            parse_yolo_label(label_text)
            manifest_entries.append(
                {
                    "image_id": image_id,
                    "split": split,
                    "source_group": image_id,
                    "image_path": image_rel.as_posix(),
                    "label_path": label_rel.as_posix(),
                    "sha256": record["sha256"],
                }
            )
            attribution_entries.append(
                {
                    "image_id": image_id,
                    "license": record["license"],
                    "author": record["author"],
                    "author_profile": record["author_profile"],
                    "title": record["title"],
                    "landing_url": record["landing_url"],
                }
            )

        class_mapping = {
            "source_dataset": "Open Images V7",
            "target_classes": [
                {
                    **entry,
                    "training_id": entry["target_id"] if entry["target_id"] < 7 else None,
                    "trainable": entry["target_id"] < 7,
                    "valid_box_count": class_box_counts[entry["target_id"]],
                    "annotated_image_count": len(class_image_counts[entry["target_id"]]),
                }
                for entry in TARGET_CLASSES
            ],
            "training_names": {entry["target_id"]: entry["name"] for entry in TRAINABLE_CLASSES},
            "bag_status": "unsupported: zero valid Open Images V7 bounding boxes",
        }
        (staging_root / "metadata").mkdir()
        (staging_root / "metadata" / "split_manifest.json").write_text(
            json.dumps(
                {
                    "split_counts": dict(split_counts),
                    "images": manifest_entries,
                },
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        (staging_root / "metadata" / "source_attributions.json").write_text(
            json.dumps(attribution_entries, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        (staging_root / "metadata" / "class_mapping.json").write_text(
            json.dumps(class_mapping, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        config = {
            "path": str(output_root),
            "train": "images/train",
            "val": "images/val",
            "test": "images/test",
            "names": {entry["target_id"]: entry["name"] for entry in TRAINABLE_CLASSES},
        }
        (staging_root / "data.yaml").write_text(
            yaml.safe_dump(config, sort_keys=False, allow_unicode=True),
            encoding="utf-8",
        )
        report = {
            "source_manifest": str(source_manifest.resolve()),
            "dataset_root": str(output_root),
            "image_counts": {split: split_counts[split] for split in SPLITS},
            "image_count": len(prepared_records),
            "valid_box_count": sum(class_box_counts.values()),
            "class_box_counts": {
                entry["name"]: class_box_counts[entry["target_id"]]
                for entry in TARGET_CLASSES
            },
            "class_image_counts": {
                entry["name"]: len(class_image_counts[entry["target_id"]])
                for entry in TARGET_CLASSES
            },
            "cross_split_exact_duplicate_groups": 0,
            "cross_split_near_duplicate_pairs_phash_le_4": 0,
            "source_label_manifest_mismatches": 0,
            "image_level_cc_by_2_attributions": len(attribution_entries),
            "untrained_classes": ["bag"],
            "box_counts_by_split": {
                split: {
                    entry["name"]: split_boxes[split][entry["target_id"]]
                    for entry in TRAINABLE_CLASSES
                }
                for split in SPLITS
            },
        }
        (staging_root / "metadata" / "preparation_report.json").write_text(
            json.dumps(report, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        staging_root.rename(output_root)
    except Exception:
        shutil.rmtree(staging_root)
        raise
    return report


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Create an isolated BYAKU YOLO dataset from the verified Open Images subset."
    )
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_ROOT)
    args = parser.parse_args()
    result = prepare_dataset(args.source_root, args.output_dir)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
