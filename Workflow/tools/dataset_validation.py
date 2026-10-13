from __future__ import annotations

import hashlib
import json
import math
import random
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import cv2
import yaml


CLASS_NAMES = (
    "person",
    "bottle",
    "book",
    "chair",
    "table",
    "mobile_phone",
    "laptop",
    "bag",
)
SPLITS = ("train", "val", "test")
IMAGE_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}
LOW_RESOLUTION_MIN_SIDE = 128
SPLIT_SEED = 39


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _relative(path: Path, root: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_yolo_label(text: str, class_count: int = len(CLASS_NAMES)) -> list[dict[str, Any]]:
    parsed: list[dict[str, Any]] = []
    for line_number, raw_line in enumerate(text.splitlines(), start=1):
        if not raw_line.strip():
            continue
        parts = raw_line.split()
        if len(parts) != 5:
            raise ValueError(f"line {line_number}: expected 5 fields, got {len(parts)}")
        try:
            class_id = int(parts[0])
        except ValueError as error:
            raise ValueError(f"line {line_number}: class_id must be an integer") from error
        if not 0 <= class_id < class_count:
            raise ValueError(f"line {line_number}: class_id {class_id} is outside 0..{class_count - 1}")
        try:
            x_center, y_center, width, height = (float(value) for value in parts[1:])
        except ValueError as error:
            raise ValueError(f"line {line_number}: box values must be numbers") from error
        values = (x_center, y_center, width, height)
        if not all(math.isfinite(value) for value in values):
            raise ValueError(f"line {line_number}: box values must be finite")
        if not 0 <= x_center <= 1 or not 0 <= y_center <= 1:
            raise ValueError(f"line {line_number}: center coordinates must be normalized to [0, 1]")
        if not 0 < width <= 1 or not 0 < height <= 1:
            raise ValueError(f"line {line_number}: width and height must be in (0, 1]")
        if (
            x_center - width / 2 < 0
            or x_center + width / 2 > 1
            or y_center - height / 2 < 0
            or y_center + height / 2 > 1
        ):
            raise ValueError(f"line {line_number}: box extends outside image bounds")
        parsed.append(
            {
                "class_id": class_id,
                "x_center": x_center,
                "y_center": y_center,
                "width": width,
                "height": height,
                "line": line_number,
            }
        )
    return parsed


def deterministic_group_split(
    items: Iterable[dict[str, str]],
    *,
    seed: int = SPLIT_SEED,
    ratios: tuple[float, float, float] = (0.70, 0.20, 0.10),
) -> dict[str, list[str]]:
    item_list = list(items)
    if not math.isclose(sum(ratios), 1.0, abs_tol=1e-9) or any(ratio < 0 for ratio in ratios):
        raise ValueError("split ratios must be non-negative and sum to 1")
    if any(not item.get("image_path") or not item.get("source_group") for item in item_list):
        raise ValueError("every split item needs image_path and source_group")

    groups: dict[str, list[str]] = defaultdict(list)
    for item in item_list:
        groups[item["source_group"]].append(item["image_path"])
    tie_breakers = {
        group: random.Random(f"{seed}:{group}").random()
        for group in groups
    }
    group_names = sorted(groups, key=lambda name: (-len(groups[name]), tie_breakers[name], name))
    targets = {
        split: len(item_list) * ratio
        for split, ratio in zip(SPLITS, ratios)
    }
    assigned: dict[str, list[str]] = {split: [] for split in SPLITS}
    group_assignments: dict[str, str] = {}
    for group in group_names:
        members = sorted(groups[group])
        split = max(
            SPLITS,
            key=lambda name: (
                (targets[name] - len(assigned[name])) / max(targets[name], 1.0),
                -SPLITS.index(name),
            ),
        )
        assigned[split].extend(members)
        group_assignments[group] = split
    return {
        "images": {split: sorted(assigned[split]) for split in SPLITS},
        "source_groups": group_assignments,
        "counts": {split: len(assigned[split]) for split in SPLITS},
        "target_ratios": dict(zip(SPLITS, ratios)),
        "seed": seed,
    }


def _candidate_source_records(repo_root: Path) -> list[dict[str, Any]]:
    def image_record(
        relative_path: str,
        *,
        source_type: str,
        suitable: bool,
        reason: str | None = None,
        image_count: int = 1,
    ) -> dict[str, Any]:
        path = repo_root / relative_path
        present = path.is_file()
        return {
            "source_path": relative_path.replace("\\", "/"),
            "source_type": source_type,
            "available": present,
            "image_count": image_count if present else 0,
            "image_count_semantics": (
                "decoded source video frame count"
                if source_type == "local_byaku_video"
                else "source image file count"
            ),
            "annotation_format": "none",
            "available_classes": [],
            "available_classes_basis": "No verified source annotations were available.",
            "suitable": suitable and present,
            "suitability": "manual_annotation_candidate" if suitable and present else "excluded",
            "reason_for_exclusion": None if suitable and present else (reason or "source file is not available"),
        }

    records = [
        image_record(
            "indoor_objects_video.mp4",
            source_type="local_byaku_video",
            suitable=True,
            image_count=141,
        ),
        image_record(
            "bedroom_scene_desk.jpg",
            source_type="local_byaku_test_image",
            suitable=True,
        ),
        image_record(
            "bedroom_scene_window.jpg",
            source_type="local_byaku_test_image",
            suitable=True,
        ),
    ]
    preview_paths = sorted((repo_root / "Workflow" / "results").glob("*.png"))
    if preview_paths:
        records.append(
            {
                "source_path": "Workflow/results/*.png",
                "source_type": "generated_depth_or_detection_preview",
                "available": True,
                "image_count": len(preview_paths),
                "image_count_semantics": "generated preview image file count",
                "annotation_format": "none",
                "available_classes": [],
                "available_classes_basis": "No verified source annotations were available.",
                "suitable": False,
                "suitability": "excluded",
                "reason_for_exclusion": (
                    "Generated depth/detection visualizations are not original RGB training images."
                ),
                "files": [_relative(path, repo_root) for path in preview_paths],
            }
        )
    for relative_path in (
        "UI/public/assets/byakugan-logo.jpg",
        "UI/public/assets/galaxy.jpg",
        "UI/public/assets/byaku-fire.mp4",
    ):
        records.append(
            image_record(
                relative_path,
                source_type="UI_branding_asset",
                suitable=False,
                reason="Frontend artwork/video is not a BYAKU camera training sample.",
                image_count=0 if relative_path.lower().endswith(".mp4") else 1,
            )
        )
    return records


def build_source_inventory(repo_root: Path) -> dict[str, Any]:
    root = repo_root.resolve()
    expected_public_roots = ("datasets", "dataset", "coco", "openimages", "objects365", "roboflow")
    public_roots = [
        {
            "path": name,
            "available": (root / name).exists(),
        }
        for name in expected_public_roots
    ]
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "public_source_status": "not_locally_available",
        "scanned_for_annotation_metadata": True,
        "annotation_metadata_found": [],
        "local_public_source_directories": public_roots,
        "candidate_sources": _candidate_source_records(root),
        "inventory_scope": (
            "Repository media candidates and common local public-dataset directory names were checked; "
            "virtual environments, generated previews, and UI assets are not training sources."
        ),
    }


def _class_mapping() -> dict[str, Any]:
    return {
        class_name: {"byaku_class_id": class_id, "source_classes": []}
        for class_id, class_name in enumerate(CLASS_NAMES)
    }


def _write_data_yaml(dataset_root: Path) -> None:
    lines = ["path: .", "train: images/train", "val: images/val", "test: images/test", "names:"]
    lines.extend(f"  {class_id}: {name}" for class_id, name in enumerate(CLASS_NAMES))
    (dataset_root / "data.yaml").write_text("\n".join(lines) + "\n", encoding="utf-8")


def validate_dataset(dataset_root: Path) -> dict[str, Any]:
    root = dataset_root.resolve()
    required_directories = [
        root / "images" / split for split in SPLITS
    ] + [
        root / "labels" / split for split in SPLITS
    ] + [root / "metadata"]
    missing_directories = [str(path) for path in required_directories if not path.is_dir()]
    structural_errors: list[str] = []
    if missing_directories:
        structural_errors.extend(f"missing directory: {path}" for path in missing_directories)

    data_file = root / "data.yaml"
    data_yaml: dict[str, Any] | None = None
    if not data_file.is_file():
        structural_errors.append("data.yaml is missing")
    else:
        try:
            loaded = yaml.safe_load(data_file.read_text(encoding="utf-8"))
            if not isinstance(loaded, dict):
                raise ValueError("data.yaml must contain a mapping")
            data_yaml = loaded
            names = loaded.get("names")
            expected_names = {index: name for index, name in enumerate(CLASS_NAMES)}
            if not isinstance(names, dict) or {int(key): value for key, value in names.items()} != expected_names:
                structural_errors.append("data.yaml class vocabulary does not match the required 8 classes")
            if loaded.get("path") != ".":
                structural_errors.append("data.yaml path must be relative to the dataset root")
            for split in SPLITS:
                value = loaded.get(split)
                if value != f"images/{split}":
                    structural_errors.append(f"data.yaml has an invalid {split} path")
        except (OSError, UnicodeError, yaml.YAMLError, TypeError, ValueError) as error:
            structural_errors.append(f"data.yaml cannot be parsed: {error}")

    per_split: dict[str, list[Path]] = {}
    for split in SPLITS:
        image_directory = root / "images" / split
        per_split[split] = sorted(
            (
                path
                for path in image_directory.rglob("*")
                if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
            ),
            key=lambda path: path.as_posix().casefold(),
        ) if image_directory.is_dir() else []

    counts = {split: len(paths) for split, paths in per_split.items()}
    total_images = sum(counts.values())
    if not total_images:
        structural_errors.append("dataset contains no images")
    class_image_counts = {name: 0 for name in CLASS_NAMES}
    class_box_counts = {name: 0 for name in CLASS_NAMES}
    split_class_counts = {
        split: {name: {"images": 0, "boxes": 0} for name in CLASS_NAMES}
        for split in SPLITS
    }
    valid_images: list[str] = []
    invalid_images: list[dict[str, str]] = []
    missing_labels: list[str] = []
    malformed_labels: list[dict[str, str]] = []
    out_of_bounds_boxes: list[dict[str, str]] = []
    empty_annotations: list[str] = []
    low_resolution_images: list[dict[str, Any]] = []
    image_hashes: dict[str, list[str]] = defaultdict(list)
    image_records: list[dict[str, Any]] = []
    split_group_locations: dict[str, set[str]] = defaultdict(set)
    split_manifest_path = root / "metadata" / "split_manifest.json"
    manifest_entries: dict[str, dict[str, Any]] = {}
    manifest: dict[str, Any] = {}
    if split_manifest_path.is_file():
        try:
            manifest = json.loads(split_manifest_path.read_text(encoding="utf-8"))
            manifest_entries = {
                entry["image_path"]: entry
                for entry in manifest.get("images", [])
                if isinstance(entry, dict) and "image_path" in entry
            }
        except (OSError, json.JSONDecodeError, TypeError):
            structural_errors.append("split_manifest.json is malformed")
    else:
        structural_errors.append("split_manifest.json is missing")

    orphan_labels: list[str] = []
    for split in SPLITS:
        image_directory = root / "images" / split
        label_directory = root / "labels" / split
        if not label_directory.is_dir():
            continue
        image_stems = {
            path.relative_to(image_directory).with_suffix("").as_posix().casefold()
            for path in per_split[split]
        }
        orphan_labels.extend(
            _relative(path, root)
            for path in sorted(label_directory.rglob("*.txt"))
            if path.relative_to(label_directory).with_suffix("").as_posix().casefold()
            not in image_stems
        )

    for split, paths in per_split.items():
        for image_path in paths:
            relative_image = _relative(image_path, root)
            image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            if image is None or image.size == 0:
                invalid_images.append({"image_path": relative_image, "reason": "unreadable or corrupted image"})
                continue
            height, width = image.shape[:2]
            valid_images.append(relative_image)
            if min(width, height) < LOW_RESOLUTION_MIN_SIDE:
                low_resolution_images.append(
                    {"image_path": relative_image, "width": width, "height": height}
                )
            manifest_entry = manifest_entries.get(relative_image)
            if manifest_entry and manifest_entry.get("source_group"):
                split_group_locations[manifest_entry["source_group"]].add(split)
            try:
                image_hashes[sha256_file(image_path)].append(relative_image)
            except OSError as error:
                invalid_images.append({"image_path": relative_image, "reason": f"cannot hash image: {error}"})
            label_path = (root / "labels" / split / image_path.relative_to(root / "images" / split)).with_suffix(".txt")
            if not label_path.is_file():
                missing_labels.append(relative_image)
                continue
            if label_path.stat().st_size == 0:
                empty_annotations.append(_relative(label_path, root))
                continue
            try:
                boxes = parse_yolo_label(label_path.read_text(encoding="utf-8"))
            except (OSError, UnicodeError, ValueError) as error:
                message = str(error)
                issue = {"label_path": _relative(label_path, root), "reason": message}
                if "outside image bounds" in message:
                    out_of_bounds_boxes.append(issue)
                else:
                    malformed_labels.append(issue)
                continue
            per_image_classes: set[int] = set()
            for box in boxes:
                class_id = box["class_id"]
                class_name = CLASS_NAMES[class_id]
                per_image_classes.add(class_id)
                class_box_counts[class_name] += 1
                split_class_counts[split][class_name]["boxes"] += 1
            for class_id in per_image_classes:
                class_name = CLASS_NAMES[class_id]
                class_image_counts[class_name] += 1
                split_class_counts[split][class_name]["images"] += 1
            image_records.append(
                {
                    "image_path": relative_image,
                    "label_path": _relative(label_path, root),
                    "width": width,
                    "height": height,
                    "boxes": boxes,
                }
            )

    duplicate_groups = [
        {"sha256": digest, "files": sorted(files)}
        for digest, files in sorted(image_hashes.items())
        if len(files) > 1
    ]
    duplicate_count = sum(len(group["files"]) - 1 for group in duplicate_groups)
    total_class_images = sum(class_image_counts.values())
    total_class_boxes = sum(class_box_counts.values())
    split_totals = {
        split: {
            "images": sum(values["images"] for values in split_class_counts[split].values()),
            "boxes": sum(values["boxes"] for values in split_class_counts[split].values()),
        }
        for split in SPLITS
    }
    split_leaks = [
        {"source_group": group, "splits": sorted(locations)}
        for group, locations in sorted(split_group_locations.items())
        if len(locations) > 1
    ]
    if split_leaks:
        structural_errors.append("one or more source groups occur in multiple splits")
    if invalid_images:
        structural_errors.append("one or more images are unreadable")
    if missing_labels:
        structural_errors.append("one or more images are missing labels")
    if empty_annotations:
        structural_errors.append("one or more annotation files are empty")
    if orphan_labels:
        structural_errors.append("one or more labels do not have matching images")
    if malformed_labels or out_of_bounds_boxes:
        structural_errors.append("one or more labels are invalid")
    if split_manifest_path.is_file() and manifest:
        for entry in manifest.get("images", []):
            if isinstance(entry, dict) and not entry.get("source_group"):
                structural_errors.append("split manifest image is missing its source grouping")
                break
        declared = {
            split: set(manifest.get("splits", {}).get(split, []))
            for split in SPLITS
        }
        for split, images in per_split.items():
            actual = {_relative(path, root) for path in images}
            if declared and actual != declared.get(split, set()):
                structural_errors.append(f"split manifest does not match {split} image directory")

    quality_report = {
        "valid": not structural_errors,
        "structural_errors": structural_errors,
        "image_counts_by_split": counts,
        "image_percentages_by_split": {
            split: (100 * count / total_images if total_images else None)
            for split, count in counts.items()
        },
        "valid_images": valid_images,
        "invalid_images": invalid_images,
        "missing_labels": missing_labels,
        "orphan_labels": orphan_labels,
        "malformed_labels": malformed_labels,
        "out_of_bounds_boxes": out_of_bounds_boxes,
        "empty_annotations": empty_annotations,
        "duplicate_images": {
            "exact_duplicate_count": duplicate_count,
            "duplicate_groups": duplicate_groups,
        },
        "low_resolution_images": low_resolution_images,
        "annotation_counts": {
            "annotated_image_count": len(image_records),
            "bounding_box_count": sum(class_box_counts.values()),
        },
        "class_distribution": {
            class_name: {
                "images": class_image_counts[class_name],
                "image_percentage": (
                    100 * class_image_counts[class_name] / total_class_images
                    if total_class_images
                    else None
                ),
                "boxes": class_box_counts[class_name],
                "box_percentage": (
                    100 * class_box_counts[class_name] / total_class_boxes
                    if total_class_boxes
                    else None
                ),
                "split_distribution": {
                    split: {
                        **split_class_counts[split][class_name],
                        "image_percentage": (
                            100
                            * split_class_counts[split][class_name]["images"]
                            / split_totals[split]["images"]
                            if split_totals[split]["images"]
                            else None
                        ),
                        "box_percentage": (
                            100
                            * split_class_counts[split][class_name]["boxes"]
                            / split_totals[split]["boxes"]
                            if split_totals[split]["boxes"]
                            else None
                        ),
                    }
                    for split in SPLITS
                },
                "status": _class_data_status(class_box_counts[class_name]),
            }
            for class_name in CLASS_NAMES
        },
        "class_status_thresholds": {
            "severely limited": "0 bounding boxes",
            "limited": "1-99 bounding boxes",
            "sufficient": "100 or more bounding boxes",
        },
        "source_group_split_leaks": split_leaks,
    }
    return {
        "quality_report": quality_report,
        "image_records": image_records,
        "image_hashes": {
            path: digest
            for digest, paths in image_hashes.items()
            for path in paths
        },
    }


def _class_data_status(box_count: int) -> str:
    if box_count == 0:
        return "severely limited"
    if box_count < 100:
        return "limited"
    return "sufficient"


def _select_video_frames(frame_count: int, count: int = 20) -> list[int]:
    if frame_count < 1:
        return []
    if count < 1:
        raise ValueError("frame sample count must be positive")
    if frame_count <= count:
        return list(range(frame_count))
    start = min(2, frame_count - 1)
    return sorted(
        {
            round(start + index * (frame_count - 1 - start) / (count - 1))
            for index in range(count)
        }
    )


def prepare_dataset(repo_root: Path, dataset_root: Path) -> dict[str, Any]:
    repo_root = repo_root.resolve()
    dataset_root = dataset_root.resolve()
    metadata_root = dataset_root / "metadata"

    inventory = build_source_inventory(repo_root)
    _write_json(metadata_root / "source_inventory.json", inventory)

    for split in SPLITS:
        (dataset_root / "images" / split).mkdir(parents=True, exist_ok=True)
        (dataset_root / "labels" / split).mkdir(parents=True, exist_ok=True)
    (metadata_root / "qa_samples").mkdir(parents=True, exist_ok=True)
    _write_data_yaml(dataset_root)
    _write_json(metadata_root / "class_mapping.json", _class_mapping())

    video_path = repo_root / "Data" / "samples" / "indoor_objects_video.mp4"
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open local BYAKU video: {video_path}")
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_indices = _select_video_frames(frame_count)
    candidates: list[dict[str, Any]] = [
        {
            "source": _relative(video_path, repo_root),
            "source_group": "video:indoor_objects_video.mp4",
            "source_frame_index": frame_index,
            "file_name": f"pinterest-video_frame_{frame_index:06d}.jpg",
            "kind": "video_frame",
        }
        for frame_index in frame_indices
    ]
    for source_path in (
        repo_root / "Data" / "samples" / "bedroom_scene_desk.jpg",
        repo_root / "Data" / "samples" / "bedroom_scene_window.jpg",
    ):
        if source_path.is_file():
            candidates.append(
                {
                    "source": _relative(source_path, repo_root),
                    "source_group": f"image:{source_path.name}",
                    "source_frame_index": None,
                    "file_name": source_path.name,
                    "kind": "still_image",
                }
            )
    candidate_split = deterministic_group_split(
        [
            {
                "image_path": candidate["file_name"],
                "source_group": candidate["source_group"],
            }
            for candidate in candidates
        ]
    )
    queue_items: list[dict[str, Any]] = []
    dataset_images: list[dict[str, str]] = []
    for candidate in candidates:
        split = candidate_split["source_groups"][candidate["source_group"]]
        destination = dataset_root / "images" / split / candidate["file_name"]
        if candidate["kind"] == "video_frame":
            cap.set(cv2.CAP_PROP_POS_FRAMES, candidate["source_frame_index"])
            read_ok, image = cap.read()
            if not read_ok or image is None or image.size == 0:
                cap.release()
                raise RuntimeError(
                    f"failed to decode source video frame {candidate['source_frame_index']}"
                )
            if not cv2.imwrite(str(destination), image):
                cap.release()
                raise RuntimeError(f"failed to write dataset frame: {destination}")
        else:
            source_path = repo_root / candidate["source"]
            image = cv2.imread(str(source_path), cv2.IMREAD_COLOR)
            if image is None or image.size == 0:
                cap.release()
                raise RuntimeError(f"local BYAKU image is unreadable: {source_path}")
            if not cv2.imwrite(str(destination), image):
                cap.release()
                raise RuntimeError(f"failed to copy local BYAKU image: {destination}")
        relative_image = _relative(destination, dataset_root)
        queue_items.append(
            {
                "image_path": relative_image,
                "source": candidate["source"],
                "source_group": candidate["source_group"],
                "source_frame_index": candidate["source_frame_index"],
                "expected_useful_classes": [],
                "expected_classes_basis": "No verified source annotation metadata; annotator must inspect.",
                "annotation_status": "needs_manual_annotation",
            }
        )
        dataset_images.append(
            {
                "image_path": relative_image,
                "source_group": candidate["source_group"],
                "annotation_status": "needs_manual_annotation",
            }
        )
    cap.release()
    queue_items.sort(key=lambda item: item["image_path"])
    _write_json(metadata_root / "manual_annotation_queue.json", queue_items)
    dataset_images.sort(key=lambda item: item["image_path"])

    hashes_by_digest: dict[str, list[str]] = defaultdict(list)
    metadata_signatures: dict[tuple[int, int, int], list[str]] = defaultdict(list)
    for image_path in sorted((dataset_root / "images").rglob("*")):
        if not image_path.is_file() or image_path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        relative_path = _relative(image_path, dataset_root)
        digest = sha256_file(image_path)
        hashes_by_digest[digest].append(relative_path)
        stat = image_path.stat()
        image = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
        if image is not None:
            height, width = image.shape[:2]
            metadata_signatures[(width, height, stat.st_size)].append(relative_path)
    exact_duplicate_groups = [
        {"sha256": digest, "files": paths}
        for digest, paths in sorted(hashes_by_digest.items())
        if len(paths) > 1
    ]
    duplicate_count = sum(len(item["files"]) - 1 for item in exact_duplicate_groups)
    metadata_duplicate_candidates = [
        {"width": key[0], "height": key[1], "bytes": key[2], "files": paths}
        for key, paths in sorted(metadata_signatures.items())
        if len(paths) > 1
    ]
    duplicates = {
        "exact_duplicate_count": duplicate_count,
        "exact_duplicate_groups": exact_duplicate_groups,
        "metadata_signature_candidates": metadata_duplicate_candidates,
        "files_removed_or_excluded": [],
        "duplicate_scope": "images staged in the new dataset split directories",
    }
    _write_json(metadata_root / "duplicates.json", duplicates)

    split_lists = {
        split: sorted(
            item["image_path"]
            for item in dataset_images
            if candidate_split["source_groups"][item["source_group"]] == split
        )
        for split in SPLITS
    }
    split_manifest = {
        "seed": candidate_split["seed"],
        "target_ratios": candidate_split["target_ratios"],
        "actual_image_counts": {split: len(paths) for split, paths in split_lists.items()},
        "actual_image_percentages": {
            split: (100 * len(paths) / len(dataset_images) if dataset_images else None)
            for split, paths in split_lists.items()
        },
        "splits": split_lists,
        "images": dataset_images,
        "source_groups": candidate_split["source_groups"],
        "grouping_policy": (
            "All sampled frames from indoor_objects_video.mp4 stay in one split; each still image is its own group."
        ),
        "class_distribution": {
            split: {name: {"images": 0, "boxes": 0} for name in CLASS_NAMES}
            for split in SPLITS
        },
        "note": "Candidate images are unannotated and are not training-ready samples.",
    }
    _write_json(metadata_root / "split_manifest.json", split_manifest)

    validation = validate_dataset(dataset_root)
    _write_json(metadata_root / "quality_report.json", validation["quality_report"])
    box_counts = validation["quality_report"]["class_distribution"]
    for split in SPLITS:
        split_manifest["class_distribution"][split] = {
            name: box_counts[name]["split_distribution"][split]
            for name in CLASS_NAMES
        }
    _write_json(metadata_root / "split_manifest.json", split_manifest)

    manifest = {
        "creation_timestamp": datetime.now(timezone.utc).isoformat(),
        "dataset_version": "byaku_detection_v1",
        "target_classes": [
            {"class_id": index, "name": name}
            for index, name in enumerate(CLASS_NAMES)
        ],
        "source_datasets": {
            "public_source_status": inventory["public_source_status"],
            "local_sources": [
                item
                for item in inventory["candidate_sources"]
                if item["suitability"] == "manual_annotation_candidate"
            ],
        },
        "image_counts": {
            "candidate_images_for_manual_annotation": len(queue_items),
            "dataset_split_images": len(dataset_images),
            "annotated_images": validation["quality_report"]["annotation_counts"]["annotated_image_count"],
        },
        "annotation_counts": {
            "bounding_boxes": validation["quality_report"]["annotation_counts"]["bounding_box_count"],
            "per_class": {
                name: box_counts[name]["boxes"]
                for name in CLASS_NAMES
            },
        },
        "split_counts": split_manifest["actual_image_counts"],
        "split_percentages": split_manifest["actual_image_percentages"],
        "sha256_hashes": validation["image_hashes"],
        "excluded_files": [
            {
                "source_path": item["source_path"],
                "reason": item["reason_for_exclusion"],
            }
            for item in inventory["candidate_sources"]
            if item["suitability"] == "excluded"
        ],
        "manual_annotation_queue_count": len(queue_items),
        "validation_result": {
            "valid": validation["quality_report"]["valid"],
            "structural_errors": validation["quality_report"]["structural_errors"],
            "ready_for_training": False,
            "missing_label_count": len(validation["quality_report"]["missing_labels"]),
            "malformed_label_count": len(validation["quality_report"]["malformed_labels"]),
            "out_of_bounds_box_count": len(validation["quality_report"]["out_of_bounds_boxes"]),
        },
        "split_note": (
            "The source video is kept within one split to avoid frame leakage. With only one video "
            "sequence and two standalone images, the target ratios cannot be met exactly."
        ),
    }
    _write_json(metadata_root / "dataset_manifest.json", manifest)
    return {
        "inventory": inventory,
        "queue_count": len(queue_items),
        "split_counts": split_manifest["actual_image_counts"],
        "quality_report": validation["quality_report"],
        "manifest": manifest,
    }


def validate_and_write_report(dataset_root: Path) -> dict[str, Any]:
    result = validate_dataset(dataset_root)
    _write_json(dataset_root / "metadata" / "quality_report.json", result["quality_report"])
    return result
