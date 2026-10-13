import json
import zipfile

import cv2
import numpy as np
import pytest

from Workflow.tools.import_visual_genome_dataset import (
    bbox_to_yolo,
    cluster_duplicate_images,
    convert_object_record,
    download_source_image,
    iter_json_array,
    iter_visual_genome_records,
    load_class_mapping,
    map_source_names,
    read_selection_jsonl,
    split_image_groups,
    stage_selected_dataset,
)


MAPPING = {
    "classes": [
        {"class_id": 0, "canonical_name": "cat", "source_labels": ["cat"]},
        {"class_id": 1, "canonical_name": "chair", "source_labels": ["chair"]},
    ]
}


def test_iter_json_array_handles_small_chunks():
    import io

    assert list(iter_json_array(io.StringIO('[{"x":1}, 2, "three"]'), chunk_size=3)) == [
        {"x": 1},
        2,
        "three",
    ]


@pytest.mark.parametrize(
    "box",
    [
        (-1, 0, 1, 1, 10, 10),
        (0, 0, 0, 1, 10, 10),
        (0, 0, 1, 0, 10, 10),
        (9, 0, 2, 1, 10, 10),
        (0, 9, 1, 2, 10, 10),
        (0, 0, float("nan"), 1, 10, 10),
        (0, 0, 1, 1, 0, 10),
    ],
)
def test_bbox_rejects_invalid_and_out_of_bounds_boxes(box):
    with pytest.raises(ValueError):
        bbox_to_yolo(*box)


def test_bbox_conversion_uses_unclipped_normalized_xywh():
    assert bbox_to_yolo(2, 1, 4, 3, 10, 10) == pytest.approx(
        (0.4, 0.25, 0.4, 0.3)
    )


def test_class_ids_are_persistent_and_source_names_are_unambiguous():
    source_to_id, class_names = load_class_mapping(MAPPING)
    assert class_names == ["cat", "chair"]
    assert source_to_id == {"cat": 0, "chair": 1}
    assert map_source_names(["CAT"], source_to_id) == ("mapped", 0, ["cat"])

    with pytest.raises(ValueError, match="contiguous"):
        load_class_mapping(
            {
                "classes": [
                    {"class_id": 1, "canonical_name": "cat", "source_labels": ["cat"]}
                ]
            }
        )


def test_unknown_and_ambiguous_source_names_are_not_silently_mapped():
    source_to_id, _ = load_class_mapping(MAPPING)
    assert map_source_names(["unmapped label"], source_to_id) == (
        "unknown",
        None,
        ["unmapped label"],
    )
    assert map_source_names(["cat", "chair"], source_to_id)[0:2] == (
        "ambiguous",
        None,
    )
    with pytest.raises(ValueError, match="ambiguous source label"):
        load_class_mapping(
            {
                "classes": [
                    {"class_id": 0, "canonical_name": "cat", "source_labels": ["shared"]},
                    {"class_id": 1, "canonical_name": "chair", "source_labels": ["shared"]},
                ]
            }
        )


def test_conversion_preserves_source_and_attribution_and_rejects_bad_boxes():
    source_to_id, _ = load_class_mapping(MAPPING)
    record = {
        "image_id": 7,
        "source_url": "https://source.example/7.jpg",
        "width": 100,
        "height": 50,
        "coco_id": 88,
        "flickr_id": 99,
        "source_metadata": {"image_id": 7, "url": "https://source.example/7.jpg"},
        "source_annotation": {
            "image_id": 7,
            "objects": [
                {
                    "object_id": 3,
                    "names": ["cat"],
                    "synsets": ["cat.n.01"],
                    "x": 10,
                    "y": 5,
                    "w": 20,
                    "h": 10,
                },
                {"object_id": 4, "names": ["chair"], "x": -1, "y": 0, "w": 4, "h": 4},
                {"object_id": 5, "names": ["unknown"], "x": 1, "y": 1, "w": 4, "h": 4},
            ],
        },
    }

    converted = convert_object_record(record, source_to_id)
    assert converted["image_id"] == 7
    assert converted["source_url"] == record["source_url"]
    assert converted["source_metadata"] == record["source_metadata"]
    assert converted["license"] == "CC-BY-4.0"
    assert converted["converted_objects"][0]["source_names"] == ["cat"]
    assert converted["converted_objects"][0]["source_synsets"] == ["cat.n.01"]
    assert converted["original_objects"] == record["source_annotation"]["objects"]
    assert converted["yolo_label_text"].startswith("0 ")
    assert [item["reason"] for item in converted["excluded_objects"]] == [
        "invalid_box",
        "unknown",
    ]


def test_duplicate_group_members_cannot_leak_across_splits():
    split = split_image_groups(
        {"image-a": "duplicate-1", "image-b": "duplicate-1", "image-c": "unique-1"},
        seed=41,
    )
    assert split["image_split"]["image-a"] == split["image_split"]["image-b"]
    assert sorted(
        image_id
        for image_ids in split["image_ids_by_split"].values()
        for image_id in image_ids
    ) == ["image-a", "image-b", "image-c"]


def test_missing_archives_and_selection_are_explicit(tmp_path):
    with pytest.raises(FileNotFoundError, match="object annotations not found"):
        next(iter_visual_genome_records(tmp_path / "objects.zip", tmp_path / "images.zip"))
    with pytest.raises(FileNotFoundError, match="selection manifest is missing"):
        next(read_selection_jsonl(tmp_path / "missing.jsonl"))


def test_archive_pairing_rejects_missing_annotation_rows(tmp_path):
    objects = tmp_path / "objects.zip"
    metadata = tmp_path / "metadata.zip"
    with zipfile.ZipFile(objects, "w") as archive:
        archive.writestr("objects.json", json.dumps([{"image_id": 1, "objects": []}]))
    with zipfile.ZipFile(metadata, "w") as archive:
        archive.writestr("image_data.json", json.dumps([]))
    with pytest.raises(ValueError, match="row counts differ"):
        list(iter_visual_genome_records(objects, metadata))


def test_archive_pairing_preserves_and_reports_conflicting_source_urls(tmp_path):
    objects = tmp_path / "objects.zip"
    metadata = tmp_path / "metadata.zip"
    with zipfile.ZipFile(objects, "w") as archive:
        archive.writestr(
            "objects.json",
            json.dumps(
                [
                    {
                        "image_id": 2,
                        "image_url": "https://cs.stanford.edu/people/rak248/VG_100K_2/2.jpg",
                        "objects": [],
                    }
                ]
            ),
        )
    with zipfile.ZipFile(metadata, "w") as archive:
        archive.writestr(
            "image_data.json",
            json.dumps(
                [
                    {
                        "image_id": 2,
                        "url": "https://cs.stanford.edu/people/rak248/VG_100K/2.jpg",
                        "width": 10,
                        "height": 10,
                    }
                ]
            ),
        )

    record = next(iter_visual_genome_records(objects, metadata))
    assert record["source_url"].endswith("/VG_100K/2.jpg")
    assert record["annotation_source_url"].endswith("/VG_100K_2/2.jpg")
    assert record["source_url_mismatch"] is True
    assert record["source_url_mismatch_same_filename"] is True


def test_image_download_requires_explicit_license_acknowledgment(tmp_path):
    with pytest.raises(PermissionError, match="acknowledgment"):
        download_source_image(
            "https://cs.stanford.edu/people/rak248/VG_100K/1.jpg",
            tmp_path / "1.jpg",
            image_id=1,
            rights_clearance=None,
        )


def test_missing_object_annotations_are_reported():
    source_to_id, _ = load_class_mapping(MAPPING)
    with pytest.raises(ValueError, match="objects annotation list"):
        convert_object_record({"image_id": 1, "width": 10, "height": 10}, source_to_id)


def test_staging_refuses_unavailable_image_instead_of_silent_fallback(tmp_path):
    row = {
        "image_id": 1,
        "source_url": "https://cs.stanford.edu/people/rak248/VG_100K/1.jpg",
        "width": 10,
        "height": 10,
        "yolo_label_text": "0 0.5 0.5 0.5 0.5\n",
    }
    with pytest.raises(FileNotFoundError, match="downloads are disabled"):
        stage_selected_dataset(
            [
                {
                    **row,
                    "rights_clearance": {
                        "status": "cleared",
                        "image_id": 1,
                        "image_url": row["source_url"],
                        "evidence_reference": "test evidence record",
                        "license": "test license",
                        "attribution": "test attribution",
                        "reviewer": "test reviewer",
                        "reviewed_at": "2026-10-10",
                    },
                }
            ],
            tmp_path / "output",
            max_images=1,
        )


def test_staging_blocks_missing_or_mismatched_image_rights_evidence(tmp_path):
    row = {
        "image_id": 5,
        "source_url": "https://cs.stanford.edu/people/rak248/VG_100K/5.jpg",
        "width": 10,
        "height": 10,
        "yolo_label_text": "0 0.5 0.5 0.5 0.5\n",
    }
    with pytest.raises(PermissionError, match="no independently reviewed"):
        stage_selected_dataset([row], tmp_path / "output", max_images=1)

    clearance = {
        "status": "cleared",
        "image_id": 5,
        "image_url": "https://cs.stanford.edu/people/rak248/VG_100K/other.jpg",
        "evidence_reference": "test evidence record",
        "license": "test license",
        "attribution": "test attribution",
        "reviewer": "test reviewer",
        "reviewed_at": "2026-10-10",
    }
    with pytest.raises(PermissionError, match="does not match"):
        stage_selected_dataset(
            [{**row, "rights_clearance": clearance}],
            tmp_path / "output",
            max_images=1,
        )


def test_staging_requires_explicit_resolution_of_source_url_mismatches(tmp_path):
    row = {
        "image_id": 2,
        "source_url": "https://cs.stanford.edu/people/rak248/VG_100K/2.jpg",
        "annotation_source_url": "https://cs.stanford.edu/people/rak248/VG_100K_2/2.jpg",
        "source_url_mismatch": True,
        "width": 10,
        "height": 10,
        "yolo_label_text": "0 0.5 0.5 0.5 0.5\n",
    }
    with pytest.raises(ValueError, match="URLs differ"):
        stage_selected_dataset([row], tmp_path / "output", max_images=1)


def test_duplicate_detection_groups_exact_duplicate_files(tmp_path):
    image = np.zeros((48, 48, 3), dtype=np.uint8)
    image[10:30, 10:30] = 255
    first = tmp_path / "first.jpg"
    duplicate = tmp_path / "duplicate.jpg"
    assert cv2.imwrite(str(first), image)
    duplicate.write_bytes(first.read_bytes())

    result = cluster_duplicate_images([first, duplicate])
    assert result["exact_duplicate_groups"]
    assert result["duplicate_group_by_path"][str(first.resolve())] == result[
        "duplicate_group_by_path"
    ][str(duplicate.resolve())]


def test_near_duplicate_grouping_keeps_different_encodings_together(tmp_path):
    image = np.zeros((48, 48, 3), dtype=np.uint8)
    image[10:30, 10:30] = 255
    png = tmp_path / "image.png"
    bmp = tmp_path / "image.bmp"
    assert cv2.imwrite(str(png), image)
    assert cv2.imwrite(str(bmp), image)

    result = cluster_duplicate_images([png, bmp])
    assert result["duplicate_group_by_path"][str(png.resolve())] == result[
        "duplicate_group_by_path"
    ][str(bmp.resolve())]
    assert result["near_duplicate_pairs"]
