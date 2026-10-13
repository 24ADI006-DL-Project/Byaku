from __future__ import annotations

import argparse
import hashlib
import heapq
import io
import json
import math
import posixpath
import re
import shutil
import urllib.request
import zipfile
from collections import Counter, defaultdict
from itertools import zip_longest
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence
from urllib.parse import urlparse

import cv2
import numpy as np


DEFAULT_SOURCE = "ranjaykrishna/visual_genome"
DEFAULT_SOURCE_REVISION = "65bc9e7e7353fff750326c9523e384701934e530"
DEFAULT_LICENSE = "CC-BY-4.0"
DEFAULT_LICENSE_URL = "https://creativecommons.org/licenses/by/4.0/"
DEFAULT_ATTRIBUTION = (
    "Visual Genome, by Ranjay Krishna et al.; "
    "https://doi.org/10.1007/s11263-016-0981-7"
)
IMAGE_URL_RE = re.compile(
    r"^https://cs\.stanford\.edu/people/rak248/(VG_100K(?:_2)?)/([0-9]+\.jpg)$"
)
SPLITS = ("train", "val", "test")
_JSON_END = object()


def _label_key(value: str) -> str:
    if not isinstance(value, str):
        raise TypeError("source labels must be strings")
    return " ".join(value.casefold().split())


def iter_json_array(stream: io.TextIOBase, *, chunk_size: int = 1 << 20) -> Iterator[Any]:
    """Incrementally decode a top-level JSON array without loading the whole file."""
    if chunk_size < 1:
        raise ValueError("chunk_size must be positive")
    decoder = json.JSONDecoder()
    buffer = ""
    position = 0
    end_of_stream = False

    def fill() -> None:
        nonlocal buffer, position, end_of_stream
        buffer = buffer[position:]
        position = 0
        chunk = stream.read(chunk_size)
        if chunk:
            buffer += chunk
        else:
            end_of_stream = True

    fill()
    while position < len(buffer) and buffer[position].isspace():
        position += 1
    if position == len(buffer) or buffer[position] != "[":
        raise ValueError("expected a top-level JSON array")
    position += 1

    while True:
        while True:
            while position < len(buffer) and (
                buffer[position].isspace() or buffer[position] == ","
            ):
                position += 1
            if position < len(buffer) and buffer[position] == "]":
                return
            try:
                item, end = decoder.raw_decode(buffer, position)
            except json.JSONDecodeError:
                if end_of_stream:
                    raise
                fill()
                continue
            position = end
            yield item
            break


def _single_json_member(archive: zipfile.ZipFile) -> str:
    names = [
        item.filename
        for item in archive.infolist()
        if not item.is_dir() and item.filename.lower().endswith(".json")
    ]
    if len(names) != 1:
        raise ValueError(f"expected exactly one JSON member in archive, found {names!r}")
    return names[0]


def iter_visual_genome_records(
    objects_archive: str | Path,
    image_metadata_archive: str | Path,
) -> Iterator[dict[str, Any]]:
    """Stream matching Visual Genome object and image metadata rows from ZIP archives."""
    objects_path = Path(objects_archive)
    metadata_path = Path(image_metadata_archive)
    if not objects_path.is_file():
        raise FileNotFoundError(f"Visual Genome object annotations not found: {objects_path}")
    if not metadata_path.is_file():
        raise FileNotFoundError(f"Visual Genome image metadata not found: {metadata_path}")
    with zipfile.ZipFile(objects_path) as objects_zip, zipfile.ZipFile(metadata_path) as metadata_zip:
        objects_member = _single_json_member(objects_zip)
        metadata_member = _single_json_member(metadata_zip)
        with (
            objects_zip.open(objects_member) as object_bytes,
            metadata_zip.open(metadata_member) as metadata_bytes,
            io.TextIOWrapper(object_bytes, encoding="utf-8") as object_text,
            io.TextIOWrapper(metadata_bytes, encoding="utf-8") as metadata_text,
        ):
            object_rows = iter_json_array(object_text)
            metadata_rows = iter_json_array(metadata_text)
            for index, pair in enumerate(
                zip_longest(object_rows, metadata_rows, fillvalue=_JSON_END)
            ):
                annotation, metadata = pair
                if annotation is _JSON_END or metadata is _JSON_END:
                    raise ValueError(
                        f"object annotation and image metadata row counts differ at index {index}"
                    )
                if not isinstance(annotation, dict) or not isinstance(metadata, dict):
                    raise ValueError(f"Visual Genome row {index} is not a JSON object")
                annotation_id = annotation.get("image_id", annotation.get("id"))
                metadata_id = metadata.get("image_id", metadata.get("id"))
                if annotation_id != metadata_id:
                    raise ValueError(
                        f"image ID mismatch at row {index}: "
                        f"annotation={annotation_id!r}, metadata={metadata_id!r}"
                    )
                if not isinstance(metadata.get("url"), str):
                    raise ValueError(f"image metadata row {index} has no source URL")
                annotation_url = annotation.get("image_url")
                if annotation_url is not None and not isinstance(annotation_url, str):
                    raise ValueError(f"annotation image_url at row {index} is not a string")
                yield {
                    "image_id": annotation_id,
                    "source_url": metadata["url"],
                    "annotation_source_url": annotation_url,
                    "source_url_mismatch": annotation_url not in (None, metadata["url"]),
                    "source_url_mismatch_same_filename": (
                        annotation_url is not None
                        and posixpath.basename(annotation_url) == posixpath.basename(metadata["url"])
                    ),
                    "width": metadata.get("width"),
                    "height": metadata.get("height"),
                    "coco_id": metadata.get("coco_id"),
                    "flickr_id": metadata.get("flickr_id"),
                    "source_metadata": metadata,
                    "source_annotation": annotation,
                }


def analyze_object_labels(
    records: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    """Count exact normalized object labels by instances and distinct images."""
    object_counts: Counter[str] = Counter()
    image_counts: Counter[str] = Counter()
    synset_counts: dict[str, Counter[str]] = defaultdict(Counter)
    raw_variants: dict[str, Counter[str]] = defaultdict(Counter)
    image_count = 0
    object_count = 0
    malformed_object_records = 0
    source_url_mismatches = 0
    mismatches_same_filename = 0
    mismatch_directions: Counter[str] = Counter()
    metadata_url_hosts: Counter[str] = Counter()
    annotation_url_hosts: Counter[str] = Counter()
    metadata_urls_supported_by_importer = 0
    mismatch_examples: list[dict[str, Any]] = []

    for record in records:
        image_count += 1
        metadata_host = urlparse(record.get("source_url") or "").netloc
        annotation_host = urlparse(record.get("annotation_source_url") or "").netloc
        metadata_url_hosts[metadata_host] += 1
        if annotation_host:
            annotation_url_hosts[annotation_host] += 1
        metadata_urls_supported_by_importer += int(
            bool(IMAGE_URL_RE.fullmatch(record.get("source_url") or ""))
        )
        if record.get("source_url_mismatch"):
            source_url_mismatches += 1
            mismatches_same_filename += int(bool(record.get("source_url_mismatch_same_filename")))
            annotation_path = urlparse(record.get("annotation_source_url") or "").path
            metadata_path = urlparse(record.get("source_url") or "").path
            annotation_segments = [part for part in annotation_path.split("/") if part]
            metadata_segments = [part for part in metadata_path.split("/") if part]
            annotation_source = "/".join(
                [annotation_host, annotation_segments[-2] if len(annotation_segments) > 1 else ""]
            )
            metadata_source = "/".join(
                [metadata_host, metadata_segments[-2] if len(metadata_segments) > 1 else ""]
            )
            direction = (
                f"{annotation_source}->{metadata_source}"
            )
            mismatch_directions[direction] += 1
            if len(mismatch_examples) < 10:
                mismatch_examples.append(
                    {
                        "image_id": record.get("image_id"),
                        "annotation_source_url": record.get("annotation_source_url"),
                        "metadata_source_url": record.get("source_url"),
                    }
                )
        annotation = record.get("source_annotation", {})
        objects = annotation.get("objects", []) if isinstance(annotation, Mapping) else []
        if not isinstance(objects, list):
            malformed_object_records += 1
            continue
        labels_on_image: set[str] = set()
        for item in objects:
            if not isinstance(item, Mapping):
                continue
            object_count += 1
            names = item.get("names")
            if not isinstance(names, list):
                continue
            normalized_names = {
                _label_key(name): name
                for name in names
                if isinstance(name, str) and name.strip()
            }
            item_labels = set(normalized_names)
            synsets = item.get("synsets")
            synset_values = (
                [synset.strip() for synset in synsets if isinstance(synset, str) and synset.strip()]
                if isinstance(synsets, list)
                else []
            )
            for key, name in normalized_names.items():
                object_counts[key] += 1
                raw_variants[key][name] += 1
                for synset in synset_values:
                    synset_counts[key][synset] += 1
            labels_on_image.update(item_labels)
        image_counts.update(labels_on_image)

    labels = {
        name: {
            "object_instances": object_counts[name],
            "images": image_counts[name],
            "raw_variants": dict(sorted(raw_variants[name].items())),
            "synsets": dict(synset_counts[name].most_common()),
        }
        for name in sorted(object_counts)
    }
    return {
        "source": DEFAULT_SOURCE,
        "source_revision": DEFAULT_SOURCE_REVISION,
        "annotation_archive": (
            "objects.json.zip (unversioned upstream URL; merged_object_ids observed; "
            "the exact release remains unresolved)"
        ),
        "image_records": image_count,
        "object_records": object_count,
        "distinct_normalized_labels": len(labels),
        "malformed_object_records": malformed_object_records,
        "annotation_metadata_url_mismatches": source_url_mismatches,
        "url_mismatches_with_same_filename": mismatches_same_filename,
        "url_mismatch_directions": dict(mismatch_directions.most_common()),
        "url_mismatch_examples": mismatch_examples,
        "metadata_url_hosts": dict(metadata_url_hosts.most_common()),
        "annotation_url_hosts": dict(annotation_url_hosts.most_common()),
        "metadata_urls_supported_by_importer": metadata_urls_supported_by_importer,
        "label_normalization": "casefold and collapse whitespace; raw source variants are preserved",
        "labels": labels,
    }


def load_class_mapping(mapping: Mapping[str, Any]) -> tuple[dict[str, int], list[str]]:
    """Load a reviewed mapping with persistent, contiguous IDs and unique source aliases."""
    classes = mapping.get("classes")
    if not isinstance(classes, list):
        raise ValueError("class mapping must contain a classes list")
    by_id: dict[int, str] = {}
    source_to_id: dict[str, int] = {}
    for entry in classes:
        if not isinstance(entry, Mapping):
            raise ValueError("every class mapping entry must be an object")
        class_id = entry.get("class_id")
        canonical_name = entry.get("canonical_name")
        source_labels = entry.get("source_labels")
        if isinstance(class_id, bool) or not isinstance(class_id, int) or class_id < 0:
            raise ValueError("class_id must be a non-negative integer")
        if class_id in by_id:
            raise ValueError(f"duplicate class_id: {class_id}")
        if not isinstance(canonical_name, str) or not canonical_name.strip():
            raise ValueError("canonical_name must be a non-empty string")
        if not isinstance(source_labels, list) or not source_labels:
            raise ValueError(f"class {canonical_name!r} must declare source_labels")
        by_id[class_id] = canonical_name
        for source_label in source_labels:
            key = _label_key(source_label)
            existing = source_to_id.get(key)
            if existing is not None and existing != class_id:
                raise ValueError(
                    f"ambiguous source label {source_label!r} maps to multiple class IDs"
                )
            source_to_id[key] = class_id
    if sorted(by_id) != list(range(len(by_id))):
        raise ValueError("class IDs must be contiguous from zero; retain IDs when extending a mapping")
    names = [by_id[index] for index in range(len(by_id))]
    if len({_label_key(name) for name in names}) != len(names):
        raise ValueError("canonical class names must be unique")
    return source_to_id, names


def map_source_names(
    names: Sequence[str] | None,
    source_to_id: Mapping[str, int],
) -> tuple[str, int | None, list[str]]:
    """Map one object label set; never silently choose between distinct mapped classes."""
    normalized = sorted({_label_key(name) for name in (names or []) if isinstance(name, str) and name.strip()})
    mapped = {source_to_id[name] for name in normalized if name in source_to_id}
    if len(mapped) > 1:
        return "ambiguous", None, normalized
    if len(mapped) == 1:
        return "mapped", next(iter(mapped)), normalized
    return "unknown", None, normalized


def bbox_to_yolo(
    x: int | float,
    y: int | float,
    width: int | float,
    height: int | float,
    image_width: int | float,
    image_height: int | float,
) -> tuple[float, float, float, float]:
    """Convert pixel xywh to normalized YOLO cxcywh without clipping invalid boxes."""
    values = (x, y, width, height, image_width, image_height)
    if any(isinstance(value, bool) for value in values):
        raise ValueError("box values and image dimensions must be finite numbers")
    try:
        left, top, box_width, box_height, frame_width, frame_height = (
            float(value) for value in values
        )
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError("box values and image dimensions must be finite numbers") from error
    if not all(math.isfinite(value) for value in (left, top, box_width, box_height, frame_width, frame_height)):
        raise ValueError("box values and image dimensions must be finite numbers")
    if frame_width <= 0 or frame_height <= 0:
        raise ValueError("image dimensions must be positive")
    if left < 0 or top < 0 or box_width <= 0 or box_height <= 0:
        raise ValueError("box coordinates must be non-negative with positive area")
    if left + box_width > frame_width or top + box_height > frame_height:
        raise ValueError("box is outside image bounds")
    return (
        (left + box_width / 2) / frame_width,
        (top + box_height / 2) / frame_height,
        box_width / frame_width,
        box_height / frame_height,
    )


def convert_object_record(
    record: Mapping[str, Any],
    source_to_id: Mapping[str, int],
) -> dict[str, Any]:
    """Convert mapped, valid source objects and preserve their original provenance."""
    width, height = record.get("width"), record.get("height")
    annotation = record.get("source_annotation")
    objects = annotation.get("objects") if isinstance(annotation, Mapping) else None
    if isinstance(width, bool) or isinstance(height, bool) or not isinstance(objects, list):
        raise ValueError("record needs image dimensions and an objects annotation list")
    labels: list[str] = []
    converted: list[dict[str, Any]] = []
    excluded: list[dict[str, Any]] = []
    for index, obj in enumerate(objects):
        if not isinstance(obj, Mapping):
            excluded.append({"object_index": index, "reason": "malformed_object"})
            continue
        status, class_id, _ = map_source_names(obj.get("names"), source_to_id)
        if status != "mapped":
            excluded.append(
                {
                    "object_index": index,
                    "reason": status,
                    "source_names": list(obj.get("names") or []),
                }
            )
            continue
        try:
            cx, cy, box_width, box_height = bbox_to_yolo(
                obj.get("x"),
                obj.get("y"),
                obj.get("w"),
                obj.get("h"),
                width,
                height,
            )
        except ValueError as error:
            excluded.append(
                {
                    "object_index": index,
                    "reason": "invalid_box",
                    "message": str(error),
                    "source_names": list(obj.get("names") or []),
                }
            )
            continue
        labels.append(
            f"{class_id} {cx:.8f} {cy:.8f} {box_width:.8f} {box_height:.8f}"
        )
        converted.append(
            {
                "class_id": class_id,
                "source_object_id": obj.get("object_id", obj.get("id")),
                "source_names": list(obj.get("names") or []),
                "source_synsets": obj.get("synsets"),
                "source_box_xywh": {
                    "x": obj.get("x"),
                    "y": obj.get("y"),
                    "width": obj.get("w"),
                    "height": obj.get("h"),
                },
                "yolo_box_cxcywh": [cx, cy, box_width, box_height],
            }
        )
    source_metadata = record.get("source_metadata")
    return {
        "image_id": record.get("image_id"),
        "source_url": record.get("source_url"),
        "annotation_source_url": record.get("annotation_source_url"),
        "source_url_mismatch": record.get("source_url_mismatch", False),
        "width": width,
        "height": height,
        "coco_id": record.get("coco_id"),
        "flickr_id": record.get("flickr_id"),
        "dataset": DEFAULT_SOURCE,
        "dataset_revision": DEFAULT_SOURCE_REVISION,
        "license": DEFAULT_LICENSE,
        "license_url": DEFAULT_LICENSE_URL,
        "attribution": DEFAULT_ATTRIBUTION,
        "source_metadata": dict(source_metadata) if isinstance(source_metadata, Mapping) else None,
        "original_objects": list(objects),
        "converted_objects": converted,
        "yolo_label_text": "\n".join(labels) + ("\n" if labels else ""),
        "excluded_objects": excluded,
    }


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def perceptual_hash(image: np.ndarray) -> int:
    if not isinstance(image, np.ndarray) or image.size == 0:
        raise ValueError("image must be a non-empty NumPy array")
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    resized = cv2.resize(gray, (32, 32), interpolation=cv2.INTER_AREA).astype(np.float32)
    dct = cv2.dct(resized)
    low = dct[:8, :8]
    threshold = float(np.median(low.flatten()[1:]))
    bits = low.flatten() > threshold
    return sum(int(bit) << index for index, bit in enumerate(bits))


def cluster_duplicate_images(
    image_paths: Sequence[str | Path],
    *,
    near_duplicate_hamming_threshold: int = 4,
) -> dict[str, Any]:
    if near_duplicate_hamming_threshold < 0:
        raise ValueError("near-duplicate threshold must be non-negative")
    if near_duplicate_hamming_threshold >= 64:
        raise ValueError("near-duplicate threshold must be less than 64")
    paths = [Path(path).resolve() for path in image_paths]
    hashes: dict[str, str] = {}
    phashes: dict[str, int] = {}
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(f"image required for duplicate analysis is missing: {path}")
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None or image.size == 0:
            raise ValueError(f"image cannot be decoded for duplicate analysis: {path}")
        key = str(path)
        hashes[key] = sha256_file(path)
        phashes[key] = perceptual_hash(image)

    parent = {str(path): str(path) for path in paths}

    def find(value: str) -> str:
        while parent[value] != value:
            parent[value] = parent[parent[value]]
            value = parent[value]
        return value

    def union(left: str, right: str) -> None:
        root_left, root_right = find(left), find(right)
        if root_left != root_right:
            first, second = sorted((root_left, root_right))
            parent[second] = first

    exact: dict[str, list[str]] = defaultdict(list)
    for path, digest in hashes.items():
        exact[digest].append(path)
    for group in exact.values():
        if len(group) > 1:
            for member in group[1:]:
                union(group[0], member)

    paths_by_phash: dict[int, list[str]] = defaultdict(list)
    for path, image_hash in phashes.items():
        paths_by_phash[image_hash].append(path)
    near_pairs: list[dict[str, Any]] = []
    for same_hash_paths in paths_by_phash.values():
        for member in same_hash_paths[1:]:
            union(same_hash_paths[0], member)
            if hashes[member] != hashes[same_hash_paths[0]]:
                near_pairs.append(
                    {
                        "image_a": same_hash_paths[0],
                        "image_b": member,
                        "hamming_distance": 0,
                    }
                )
    unique_hashes = sorted(paths_by_phash)
    band_count = near_duplicate_hamming_threshold + 1
    base_width, remainder = divmod(64, band_count)
    bands: list[tuple[int, int]] = []
    offset = 0
    for band in range(band_count):
        width = base_width + int(band < remainder)
        bands.append((offset, width))
        offset += width
    bucket_members: dict[tuple[int, int], list[int]] = defaultdict(list)
    for image_hash in unique_hashes:
        for band, (offset, width) in enumerate(bands):
            value = (image_hash >> offset) & ((1 << width) - 1)
            bucket_members[(band, value)].append(image_hash)
    candidate_pairs: set[tuple[int, int]] = set()
    for members in bucket_members.values():
        for index, left_hash in enumerate(members):
            for right_hash in members[index + 1 :]:
                candidate_pairs.add((left_hash, right_hash))

    for left_hash, right_hash in sorted(candidate_pairs):
        distance = (left_hash ^ right_hash).bit_count()
        if distance <= near_duplicate_hamming_threshold:
            left, right = paths_by_phash[left_hash][0], paths_by_phash[right_hash][0]
            union(left, right)
            near_pairs.append(
                {"image_a": left, "image_b": right, "hamming_distance": distance}
            )
    keys = sorted(phashes)
    groups: dict[str, list[str]] = defaultdict(list)
    for key in keys:
        groups[find(key)].append(key)
    return {
        "sha256_by_path": hashes,
        "exact_duplicate_groups": [
            sorted(paths) for paths in exact.values() if len(paths) > 1
        ],
        "near_duplicate_pairs": near_pairs,
        "duplicate_group_by_path": {
            path: min(groups[find(path)])
            for path in keys
        },
    }


def split_image_groups(
    image_to_group: Mapping[str, str],
    *,
    seed: int = 41,
    ratios: tuple[float, float, float] = (0.8, 0.1, 0.1),
) -> dict[str, Any]:
    if len(ratios) != len(SPLITS) or any(value < 0 for value in ratios):
        raise ValueError("three non-negative split ratios are required")
    if not math.isclose(sum(ratios), 1.0, abs_tol=1e-9):
        raise ValueError("split ratios must sum to 1")
    groups: dict[str, list[str]] = defaultdict(list)
    for image_id, group_id in image_to_group.items():
        if not image_id or not group_id:
            raise ValueError("image IDs and duplicate group IDs must be non-empty")
        groups[group_id].append(image_id)
    total = len(image_to_group)
    targets = {
        split: total * ratio
        for split, ratio in zip(SPLITS, ratios)
    }
    assigned: dict[str, list[str]] = {split: [] for split in SPLITS}
    group_split: dict[str, str] = {}
    ordered_groups = sorted(
        groups,
        key=lambda group: (
            -len(groups[group]),
            hashlib.sha256(f"{seed}:{group}".encode("utf-8")).hexdigest(),
        ),
    )
    for group in ordered_groups:
        split = max(
            SPLITS,
            key=lambda candidate: (
                (targets[candidate] - len(assigned[candidate]))
                / max(targets[candidate], 1.0),
                -SPLITS.index(candidate),
            ),
        )
        group_split[group] = split
        assigned[split].extend(groups[group])
    return {
        "seed": seed,
        "ratios": dict(zip(SPLITS, ratios)),
        "image_ids_by_split": {
            split: sorted(image_ids)
            for split, image_ids in assigned.items()
        },
        "duplicate_group_split": group_split,
        "image_split": {
            image_id: group_split[group_id]
            for image_id, group_id in image_to_group.items()
        },
    }


def select_annotation_records(
    records: Iterable[Mapping[str, Any]],
    source_to_id: Mapping[str, int],
    *,
    max_images: int,
) -> list[dict[str, Any]]:
    """Select a reproducible image-level sample, prioritizing relevant valid annotations."""
    if isinstance(max_images, bool) or not isinstance(max_images, int) or max_images < 1:
        raise ValueError("max_images must be a positive integer")
    selected: list[tuple[int, int, dict[str, Any]]] = []
    for record in records:
        converted = convert_object_record(record, source_to_id)
        if not converted["converted_objects"]:
            continue
        image_id = converted.get("image_id")
        rank = int(hashlib.sha256(str(image_id).encode("utf-8")).hexdigest(), 16)
        entry = (-rank, int(image_id) if isinstance(image_id, int) else rank, converted)
        if len(selected) < max_images:
            heapq.heappush(selected, entry)
        elif entry > selected[0]:
            heapq.heapreplace(selected, entry)
    result = [item[2] for item in selected]
    return sorted(result, key=lambda item: str(item["image_id"]))


def write_selection_jsonl(records: Iterable[Mapping[str, Any]], path: str | Path) -> int:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with target.open("w", encoding="utf-8", newline="\n") as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
            count += 1
    return count


def _validate_image_rights_clearance(
    image_id: Any,
    image_url: str,
    clearance: Mapping[str, Any] | None,
) -> dict[str, Any]:
    required_fields = (
        "evidence_reference",
        "license",
        "attribution",
        "reviewer",
        "reviewed_at",
    )
    if not isinstance(clearance, Mapping) or clearance.get("status") != "cleared":
        raise PermissionError(
            f"image {image_id!r} has no independently reviewed image-level rights clearance"
        )
    if clearance.get("image_id") != image_id or clearance.get("image_url") != image_url:
        raise PermissionError(
            f"image {image_id!r} rights clearance does not match the exact source image"
        )
    missing = [
        field for field in required_fields
        if not isinstance(clearance.get(field), str) or not clearance[field].strip()
    ]
    if missing:
        raise PermissionError(
            f"image {image_id!r} rights clearance is missing required evidence: {missing!r}"
        )
    return dict(clearance)


def read_selection_jsonl(path: str | Path) -> Iterator[dict[str, Any]]:
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"selection manifest is missing: {source}")
    with source.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"invalid JSON at {source}:{line_number}: {error}") from error
            if not isinstance(record, dict):
                raise ValueError(f"selection row {line_number} must be a JSON object")
            yield record


def stage_selected_dataset(
    records: Iterable[Mapping[str, Any]],
    output_root: str | Path,
    *,
    image_cache: str | Path | None = None,
    download_images: bool = False,
    acknowledge_cc_by_attribution: bool = False,
    use_image_metadata_url_on_mismatch: bool = False,
    max_images: int | None = None,
) -> dict[str, Any]:
    """Stage already-selected annotations, hash/cluster images, then split by image group."""
    if download_images and not acknowledge_cc_by_attribution:
        raise PermissionError(
            "image staging from URLs requires explicit CC-BY-4.0 attribution acknowledgment"
        )
    rows = list(records)
    if not rows:
        raise ValueError("selection contains no images to stage")
    if (
        isinstance(max_images, bool)
        or not isinstance(max_images, int)
        or max_images < 1
    ):
        raise ValueError("a positive max_images cap is required for staging")
    if len(rows) > max_images:
        raise ValueError(f"selection has {len(rows)} images, over the max_images cap of {max_images}")
    mismatched_urls = [
        record.get("image_id")
        for record in rows
        if record.get("source_url_mismatch")
    ]
    if mismatched_urls and not use_image_metadata_url_on_mismatch:
        raise ValueError(
            "annotation and image-metadata URLs differ for selected images "
            f"(for example {mismatched_urls[:5]!r}); explicitly review and enable "
            "use_image_metadata_url_on_mismatch to use metadata URLs"
        )
    rights_clearances: dict[str, dict[str, Any]] = {}
    for record in rows:
        image_id = record.get("image_id")
        url = record.get("source_url")
        if not isinstance(url, str):
            raise ValueError(f"image {image_id!r} has no metadata source URL")
        rights_clearances[str(image_id)] = _validate_image_rights_clearance(
            image_id,
            url,
            record.get("rights_clearance"),
        )
    ids = [str(record.get("image_id")) for record in rows]
    if len(set(ids)) != len(ids) or any(value in ("", "None") for value in ids):
        raise ValueError("selection must contain unique, non-empty image IDs")
    root = Path(output_root)
    cache = Path(image_cache) if image_cache is not None else None
    staging_dir = root / ".staging_images"
    image_paths: dict[str, Path] = {}
    converted_records: dict[str, dict[str, Any]] = {}
    for record in rows:
        image_id = str(record["image_id"])
        url = record.get("source_url")
        match = IMAGE_URL_RE.fullmatch(url) if isinstance(url, str) else None
        if match is None:
            raise ValueError(f"invalid Visual Genome source URL for image {image_id}")
        filename = match.group(2)
        candidates = (
            [cache / filename, cache / match.group(1) / filename]
            if cache is not None
            else []
        )
        local = next((path for path in candidates if path.is_file()), None)
        if local is not None:
            image_path = local
        elif download_images:
            image_path = staging_dir / filename
            download_source_image(
                url,
                image_path,
                image_id=record["image_id"],
                rights_clearance=rights_clearances[image_id],
                acknowledge_cc_by_attribution=acknowledge_cc_by_attribution,
            )
        else:
            raise FileNotFoundError(
                f"source image {filename} is not in the local cache and downloads are disabled"
            )
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None or image.size == 0:
            raise ValueError(f"source image cannot be decoded: {image_path}")
        expected_width = record.get("width")
        expected_height = record.get("height")
        if image.shape[1] != expected_width or image.shape[0] != expected_height:
            raise ValueError(
                f"image {image_id} dimensions differ from source metadata: "
                f"actual={image.shape[1]}x{image.shape[0]}, "
                f"metadata={expected_width}x{expected_height}"
            )
        labels = record.get("yolo_label_text")
        if not isinstance(labels, str) or not labels.strip():
            raise ValueError(f"image {image_id} has no valid converted YOLO labels")
        image_paths[image_id] = image_path
        converted_records[image_id] = dict(record)

    duplicates = cluster_duplicate_images(list(image_paths.values()))
    image_to_group = {
        image_id: duplicates["duplicate_group_by_path"][str(path.resolve())]
        for image_id, path in image_paths.items()
    }
    split_result = split_image_groups(image_to_group)
    split_by_id = split_result["image_split"]
    manifest_rows = []
    for image_id, source_path in image_paths.items():
        split = split_by_id[image_id]
        output_image = root / "images" / split / source_path.name
        output_label = root / "labels" / split / f"{Path(source_path.name).stem}.txt"
        output_source = root / "source_annotations" / split / f"{Path(source_path.name).stem}.json"
        output_image.parent.mkdir(parents=True, exist_ok=True)
        output_label.parent.mkdir(parents=True, exist_ok=True)
        output_source.parent.mkdir(parents=True, exist_ok=True)
        if source_path.resolve() != output_image.resolve():
            shutil.copy2(source_path, output_image)
        output_label.write_text(converted_records[image_id]["yolo_label_text"], encoding="utf-8")
        original_objects = converted_records[image_id].get("original_objects")
        source_payload = {
            "image_id": converted_records[image_id]["image_id"],
            "source_url": converted_records[image_id]["source_url"],
            "annotation_source_url": converted_records[image_id].get("annotation_source_url"),
            "source_url_mismatch": converted_records[image_id].get(
                "source_url_mismatch", False
            ),
            "coco_id": converted_records[image_id].get("coco_id"),
            "flickr_id": converted_records[image_id].get("flickr_id"),
            "dataset": converted_records[image_id].get("dataset", DEFAULT_SOURCE),
            "dataset_revision": converted_records[image_id].get(
                "dataset_revision", DEFAULT_SOURCE_REVISION
            ),
            "license": converted_records[image_id].get("license", DEFAULT_LICENSE),
            "license_url": converted_records[image_id].get("license_url", DEFAULT_LICENSE_URL),
            "attribution": converted_records[image_id].get("attribution", DEFAULT_ATTRIBUTION),
            "rights_clearance": rights_clearances[image_id],
            "source_metadata": converted_records[image_id].get("source_metadata"),
            "original_objects": original_objects,
            "converted_objects": converted_records[image_id].get("converted_objects"),
            "excluded_objects": converted_records[image_id].get("excluded_objects"),
            "sha256": duplicates["sha256_by_path"][str(source_path.resolve())],
            "duplicate_group": image_to_group[image_id],
            "split": split,
        }
        _write_json(output_source, source_payload)
        manifest_rows.append(
            {
                "image_id": image_id,
                "source_url": converted_records[image_id]["source_url"],
                "split": split,
                "image_path": str(output_image.relative_to(root)),
                "label_path": str(output_label.relative_to(root)),
                "source_annotation_path": str(output_source.relative_to(root)),
                "sha256": source_payload["sha256"],
                "duplicate_group": image_to_group[image_id],
                "converted_object_count": len(source_payload["converted_objects"] or []),
                "excluded_object_count": len(source_payload["excluded_objects"] or []),
            }
        )
    manifest = {
        "dataset": DEFAULT_SOURCE,
        "dataset_revision": DEFAULT_SOURCE_REVISION,
        "license": DEFAULT_LICENSE,
        "license_url": DEFAULT_LICENSE_URL,
        "attribution": DEFAULT_ATTRIBUTION,
        "split_seed": split_result["seed"],
        "split_ratios": split_result["ratios"],
        "image_count": len(rows),
        "images_by_split": {
            split: len(split_result["image_ids_by_split"][split])
            for split in SPLITS
        },
        "near_duplicate_hamming_threshold": 4,
        "exact_duplicate_groups": duplicates["exact_duplicate_groups"],
        "near_duplicate_pairs": duplicates["near_duplicate_pairs"],
        "images": sorted(manifest_rows, key=lambda item: item["image_id"]),
    }
    _write_json(root / "dataset_manifest.json", manifest)
    if staging_dir.exists() and not any(staging_dir.iterdir()):
        staging_dir.rmdir()
    return manifest


def download_source_image(
    url: str,
    destination: str | Path,
    *,
    image_id: Any,
    rights_clearance: Mapping[str, Any] | None,
    acknowledge_cc_by_attribution: bool = False,
    max_bytes: int = 30 * 1024 * 1024,
    timeout_seconds: float = 30,
) -> int:
    """Download one known Visual Genome JPEG only after explicit license acknowledgment."""
    if not acknowledge_cc_by_attribution:
        raise PermissionError(
            "explicit CC-BY-4.0 attribution acknowledgment is required before image download"
        )
    _validate_image_rights_clearance(image_id, url, rights_clearance)
    if not isinstance(url, str) or not IMAGE_URL_RE.fullmatch(url):
        raise ValueError("image URL is not an approved Visual Genome image URL")
    if max_bytes < 1 or timeout_seconds <= 0:
        raise ValueError("download limits must be positive")
    path = Path(destination)
    path.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(url, headers={"User-Agent": "BYAKU-VisualGenome-Importer/1.0"})
    total = 0
    try:
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
            content_type = response.headers.get_content_type()
            if content_type != "image/jpeg":
                raise ValueError(f"expected image/jpeg, received {content_type}")
            with path.open("wb") as target:
                while chunk := response.read(1024 * 1024):
                    total += len(chunk)
                    if total > max_bytes:
                        raise ValueError(f"image exceeds configured {max_bytes}-byte limit")
                    target.write(chunk)
    except Exception:
        path.unlink(missing_ok=True)
        raise
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None or image.size == 0:
        path.unlink(missing_ok=True)
        raise ValueError(f"downloaded image is unreadable: {url}")
    return total


def _load_mapping_file(path: Path) -> tuple[dict[str, int], list[str]]:
    if not path.is_file():
        raise FileNotFoundError(f"reviewed class mapping file is missing: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError("class mapping JSON must contain an object")
    return load_class_mapping(value)


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def analyze_archives(objects_archive: Path, metadata_archive: Path) -> dict[str, Any]:
    return analyze_object_labels(iter_visual_genome_records(objects_archive, metadata_archive))


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Analyze Visual Genome object annotations or convert a reviewed selection."
    )
    parser.add_argument("--objects-archive", type=Path, required=True)
    parser.add_argument("--image-metadata-archive", type=Path, required=True)
    parser.add_argument("--analysis-output", type=Path)
    parser.add_argument("--class-mapping", type=Path)
    parser.add_argument("--selection-output", type=Path)
    parser.add_argument("--stage-selection", type=Path)
    parser.add_argument("--image-cache", type=Path)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--download-images", action="store_true")
    parser.add_argument("--acknowledge-cc-by-attribution", action="store_true")
    parser.add_argument("--use-image-metadata-url-on-mismatch", action="store_true")
    parser.add_argument("--max-images", type=int)
    args = parser.parse_args()
    if args.analysis_output is not None:
        _write_json(
            args.analysis_output,
            analyze_archives(args.objects_archive, args.image_metadata_archive),
        )
    if args.download_images and not args.acknowledge_cc_by_attribution:
        parser.error("--download-images requires --acknowledge-cc-by-attribution")
    if args.max_images is not None and args.max_images < 1:
        parser.error("--max-images must be positive")
    if args.selection_output is not None:
        if args.class_mapping is None:
            parser.error("--selection-output requires --class-mapping")
        if args.max_images is None:
            parser.error("--selection-output requires --max-images")
        source_to_id, _ = _load_mapping_file(args.class_mapping)
        selected = select_annotation_records(
            iter_visual_genome_records(args.objects_archive, args.image_metadata_archive),
            source_to_id,
            max_images=args.max_images,
        )
        write_selection_jsonl(selected, args.selection_output)
    if args.stage_selection is not None:
        if args.output_root is None:
            parser.error("--stage-selection requires --output-root")
        if args.max_images is None:
            parser.error("--stage-selection requires --max-images")
        manifest = stage_selected_dataset(
            read_selection_jsonl(args.stage_selection),
            args.output_root,
            image_cache=args.image_cache,
            download_images=args.download_images,
            acknowledge_cc_by_attribution=args.acknowledge_cc_by_attribution,
            use_image_metadata_url_on_mismatch=args.use_image_metadata_url_on_mismatch,
            max_images=args.max_images,
        )
        print(json.dumps({"image_count": manifest["image_count"], "images_by_split": manifest["images_by_split"]}))
    if args.class_mapping is not None and args.selection_output is None:
        parser.error("--class-mapping requires --selection-output")
    if args.stage_selection is None and args.selection_output is None and args.analysis_output is None:
        parser.error(
            "provide --analysis-output, --selection-output, or --stage-selection"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
