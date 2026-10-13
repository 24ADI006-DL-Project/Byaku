# BYAKU — Phase 2 Legal Data Sources Investigation Report

## 1. Overview and Objective

To build a high-quality, legally compliant object detector for the eight BYAKU target classes:
1. `person`
2. `bottle`
3. `book`
4. `chair`
5. `table`
6. `mobile_phone`
7. `laptop`
8. `bag`

We systematically evaluated three candidate sources:
- **Open Images V7** ([https://storage.googleapis.com/openimages/web/download_v7.html](https://storage.googleapis.com/openimages/web/download_v7.html))
- **COCO** ([https://cocodataset.org/](https://cocodataset.org/))
- **Roboflow Universe** ([https://universe.roboflow.com/](https://universe.roboflow.com/))

Evaluation criteria included: class coverage, annotation schema, image reuse terms and individual creator attribution, download mechanism, dataset size, and YOLO compatibility.

---

## 2. Source Comparison

| Dimension | Open Images V7 | COCO (2017 / COCO128) | Roboflow Universe |
| :--- | :--- | :--- | :--- |
| **Target Class Coverage** | **7 of 8 classes** fully covered with tens of thousands of instances. Class `bag` (`/m/0n5v01m`) has 0 boxes in Open Images (image-level label only). | **5 of 8 exact** (`person`, `bottle`, `chair`, `book`, `laptop`). `cell phone` maps to `mobile_phone`. `dining table` is narrower than `table`. `backpack` and `handbag` exist, but no generic `bag`. | Highly fragmented across user projects; various bag datasets exist under varied naming. |
| **Annotation Quality** | Extremely high; 90% manually annotated via extreme-clicking (`xclick`), human-verified IoU > 0.7. | High; polygon-derived bounding boxes, human-verified. | Highly variable; crowd-sourced / personal projects with inconsistent box tightness. |
| **Annotation License** | Creative Commons Attribution 4.0 International (**CC BY 4.0**). | Creative Commons Attribution 4.0 International (**CC BY 4.0**). | Varied per project (often CC BY 4.0 or CC0 stated). |
| **Image Reuse Terms** | **Verified per-image Creative Commons licenses** (CC BY 2.0, CC BY-SA 2.0, CC0) explicitly recorded in official metadata CSVs. | **Unresolved per-image**. COCO Consortium explicitly disclaims copyright ownership; Flickr creator/license records are not packaged in standard downloads. | **Unverified provenance**. While project author selects CC BY 4.0, original image sources are frequently web-scraped without verifiable creator consent. |
| **Attribution Feasibility** | **Full metadata provided**: `Author`, `AuthorProfileURL`, `Title`, `OriginalLandingURL`, and `License` URL for every individual image ID. | **Infeasible** without extensive external Flickr scraping/re-identification. | Infeasible; individual photo provenance generally not tracked. |
| **Download Mechanism** | On-demand image downloads from official CVDF AWS S3 mirror (`open-images-dataset.s3.amazonaws.com/{split}/{image_id}.jpg`). | Monolithic archives (train ~19 GB, val ~1 GB) or Ultralytics COCO128 zip. | Requires Roboflow API key and account; Cloudflare blocks direct automated requests (HTTP 403). |
| **YOLO Compatibility** | Native normalized `[0.0, 1.0]` coordinates convert directly to YOLO format (`x_center, y_center, w, h`). | Pixel coordinates in JSON require conversion; COCO128 is pre-converted to YOLO format. | Typically pre-exported in YOLO format when downloaded via API. |

---

## 3. Findings and Decision

1. **Primary Dataset Source**:
   **Open Images V7** is selected as the primary source for the BYAKU dataset. It is the only repository among the candidates that provides:
   - Complete, individual per-image Creative Commons license records.
   - Verifiable photographer credit and Flickr URLs.
   - High-precision, extreme-click bounding boxes.
   - Lightweight, targeted S3 image downloads that do not require multi-gigabyte bulk downloads.
2. **COCO Decision**:
   COCO remains **disqualified** as established in `coco128_validation_report.md` because its image copyright terms require Flickr compliance, yet individual image attribution metadata is omitted from the official package.
3. **Roboflow Universe Decision**:
   Roboflow Universe is **disqualified** for the core dataset due to unverified underlying image rights and automated endpoint access restrictions (Cloudflare 403).
4. **Target Class `bag` Shortage Finding**:
   In Open Images V7, `/m/0n5v01m` is an image-level label and possesses 0 bounding box annotations across all splits. In adherence to strict project guidelines ("Do not invent annotations or merge classes silently", "If a class lacks sufficient valid samples, report the shortage and investigate another source rather than inventing annotations"):
   - We document that `bag` has 0 samples in Open Images V7.
   - We retain class `bag` in `data.yaml` at index 7 with 0 boxes documented.
   - The remaining 7 classes (`person`, `bottle`, `book`, `chair`, `table`, `mobile_phone`, `laptop`) have verified, abundant annotations.
