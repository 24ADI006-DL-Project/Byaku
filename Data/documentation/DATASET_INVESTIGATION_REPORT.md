# BYAKU — Phase 1 Investigation Report: Dataset State & Feasibility

## 1. Executive Summary

This report completes **Phase 1** of the custom YOLO11n object detector pipeline for BYAKU.
We inspected the existing environment, evaluated dataset candidates across `open_images_v7`, `coco128`, and `visual_genome`, diagnosed why previous datasets remained small or blocked, verified class hierarchy representations, and evaluated system compute feasibility.

---

## 2. Current State of `Data/datasets/open_images_v7/`

The initial Open Images V7 preparation script completed all steps for an initial 300-image pilot:
- **Download and Verification**: 300 / 300 images downloaded from the official CVDF S3 mirror (`open-images-dataset.s3.amazonaws.com`).
- **Integrity**: 100% of images verified with Pillow (`Image.verify()`).
- **Splits**: 
  - Train: 210 images (70%)
  - Validation: 60 images (20%)
  - Test: 30 images (10%)
- **Split Leakage**: 0 overlapping images across splits (strictly partitioned by image ID).
- **Labels**: 1,159 valid normalized bounding boxes formatted into standard YOLO format (`class_id x_center y_center width height`).

### Current Per-Class Counts in `byaku_yolo_v7/`

| Class Name | Open Images MID | YOLO ID | Train Boxes | Val Boxes | Test Boxes | Total Boxes | Images Containing Class |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **person** | `/m/01g317` | 0 | 197 | 66 | 22 | 285 | 109 |
| **bottle** | `/m/04dr76w` | 1 | 118 | 30 | 11 | 159 | 43 |
| **book** | `/m/0bt_c3` | 2 | 153 | 47 | 109 | 309 | 42 |
| **chair** | `/m/01mzpv` | 3 | 102 | 42 | 6 | 150 | 59 |
| **table** | `/m/04bcr3` | 4 | 104 | 25 | 13 | 142 | 73 |
| **mobile_phone** | `/m/050k8` | 5 | 48 | 11 | 4 | 63 | 44 |
| **laptop** | `/m/01c648` | 6 | 36 | 12 | 3 | 51 | 42 |
| **bag** | `/m/0n5v01m` | 7 | 0 | 0 | 0 | 0 | 0 |
| **TOTAL** | — | — | **758** | **233** | **168** | **1,159** | **300** |

---

## 3. Root Causes: Why Previous Datasets Were Too Small or Blocked

1. **Explicit Pilot Cap in Script**:
   The initial pilot script `dataset_pipeline.py` intentionally targeted `max_images = 300` to validate the ingestion and download pipeline before expanding.
2. **Open Images V7 Class Hierarchy Mismatch for `Bag`**:
   While the other 7 classes are part of the official 600 boxable classes (`oidv7-class-descriptions-boxable.csv` and `bbox_labels_600_hierarchy.json`), `/m/0n5v01m` ("Bag") is an *image-level classification label* across the 20,638 Open Images classes. It has **0 bounding boxes** in Open Images V7. Boxable bag classes in Open Images are specialized categories like Handbag (`/m/080hkjn`), Luggage and bags (`/m/0hf58v5`), or Plastic bag (`/m/05gqfk`). Per strict instructions ("Do not invent annotations or merge classes silently"), no silent remapping was performed.
3. **Legal / Attribution Stop Conditions in Prior Imports**:
   - `Visual Genome` (Step 43): Blocked at legal gate because per-image rights/creator credits were missing, and 83,139 image URLs were mismatched.
   - `COCO128` (Step 44): Blocked at legal gate because the Ultralytics zip archive omitted per-image Flickr license and creator attribution metadata. Furthermore, COCO lacks a general `bag` class (only `backpack` and `handbag`) and only has `dining table` which is narrower than `table`.

---

## 4. Hardware, System Resources & Dependencies

- **Operating System**: Windows 11 Home (AMD64)
- **CPU**: 12th Gen Intel Core i5-1235U (10 cores, 12 threads)
- **System Memory**: 15.7 GB total, 3.6–4.5 GB available
- **Storage**: C: Drive has **106.2 GB free space** (abundantly sufficient)
- **PyTorch**: `2.14.0+cpu`
- **CUDA Availability**: **False** (CPU execution only; no active NVIDIA GPU)
- **Ultralytics**: Version `8.4.166` installed in `.venv-1`
- **Base Checkpoint**: `yolo11n.pt` located at repository root (5,613,764 bytes, SHA-256 `0EBBC80D4A7680D14987A577CD21342B65ECFD94632BD9A8DA63AE6417644EE1`)
- **Implications for Training**: Because execution is CPU-only, fine-tuning must be configured conservatively:
  - Batch size: 8 to 16
  - Number of workers: 2
  - Image size: 640
  - Epochs: 20–30 with early stopping to prevent excessive runtimes while providing solid convergence.

---

## 5. Existing Training Scripts and Isolation

- One-time training and evaluation write-ups were not retained; model checkpoints and structured run metadata remain under `Models/training_runs/`. Training was deferred in Step 44 before the later recorded runs.
- The production detector in `detection/` and `app.py` uses root `yolo11n.pt`. All new training runs must write strictly to a separate directory under `Models/training_runs/` to avoid any modification of production assets.
