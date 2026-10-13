# BYAKU Dataset and Training Workflow Audit

Audit performed 2026-10-11 before creating a new prepared dataset or training run.
Counts below distinguish current filesystem evidence from inherited reports.

## Inventory

| Source | Downloaded / present | Extracted or staged images | Annotation evidence | Validated | Trainable now |
|---|---:|---:|---|---|---|
| Open Images V7 raw image pool | 759 JPEG files under `open_images_v7/raw_images`; 32,320 licensed candidate images are claimed by `raw/phase2_candidate_summary.json`, but only 600 are represented in the selected manifest | 600 selected images in `byaku_yolo_v7` (420 train, 120 val, 60 test); 159 additional raw files are not in the selected manifest and are not admitted | 600 matched labels, 2,167 boxes from the full manifest; seven target classes have boxes, bag has zero | All 600 images readable; labels match manifest boxes; no malformed IDs/values; no exact duplicates or cross-split pHash pairs at Hamming distance <=4 | 600 for a seven-class experiment: 420 train, 120 validation, 60 held-out test. Bag excluded from model outputs |
| Visual Genome | Two annotation ZIP archives, both CRC-valid; no image download is evidenced | Two annotation JSON files extracted; no image files staged | Existing import report records 108,077 source records and 88,494 mapped-box candidates; zero imported YOLO annotations | Archive integrity verified; per-image image-use rights/attribution remain unverified | 0 |
| COCO128 | `coco128.zip` present (263 archive members) | 128 JPEGs decoded in memory; no extracted training images | 128 label files and 941 source rows; existing review rejects 42 boxes extending outside image bounds | All 128 images decode and ZIP CRC check passes; archive lacks per-image rights/attribution required by the existing gate | 0 |
| Local BYAKU candidate images | Local media, not a licensed public dataset | 22 images in `byaku_detection` (20 train, 1 val, 1 test) | 0 boxes; all 22 images lack label files | Readability only; no annotations or image-use clearance established | 0 |

## Open Images verification

- The full selected manifest and attribution ledger each contain 600 distinct IDs.
- Every manifest entry points to an existing JPEG and label. The full dataset split is 420/120/60.
- Every selected record has a CC BY 2.0 license URL, creator, title, and source landing URL.
- The 2,167 manifest boxes and the current YOLO labels agree class-by-class and box-by-box within `2e-5`; all IDs are 0-6. Seven target classes have support; canonical target `bag` (ID 7) has zero boxes.
- The extents of some current six-decimal YOLO boxes exceed an image border by less than `1e-6` due to serialization rounding. Re-generated labels in the prepared copy use 12 decimal places and pass the repository's strict parser; source labels are not changed.
- SHA-256 found zero duplicate files. A 64-bit DCT perceptual hash found zero cross-split pairs with Hamming distance <=4.
- The raw image directory contains 159 files that are not identified in the selected 600-image manifest. Their annotation/license linkage is not established by the selected manifest; they are excluded.
- A new isolated dataset was prepared at `dataset/prepared_open_images_v7_full600`: 600 copied images, 2,167 manifest-derived labels, 420/120/60 split, seven training classes and preserved per-image attribution. The prepared labels pass the strict YOLO parser. The original `open_images_v7` source tree remains unchanged.
- Training class IDs remain stable for person through laptop (0-6). Canonical bag retains target ID 7 in the mapping but has no training ID because it has zero boxes; no automatic proposals or synthetic labels were created.

## Conflicting or missing evidence

- The removed `validation_report.md` and earlier 300-image manifest described the pilot (210/60/30), while the current full manifest and quality-gate report describe 600 images (420/120/60). The full manifest is the current filesystem record.
- The full manifest and `results.csv` show 2,167 boxes, including 1,585 train boxes. The training report's per-split text says 1,479 train boxes; that number does not match the current labels.
- Existing reports claim a prior 19-epoch run and evaluation. The report artifacts and current checkpoint files exist, but their metrics are not treated as new measurements in this audit.
- Existing Open Images candidate summaries describe a larger candidate pool; no complete licensed-and-annotated larger split is present locally. No download is needed for the available 600-image prepared subset.
- Visual Genome image URLs and individual image rights were not verified. COCO's official terms state the consortium does not own the images and image use is governed by Flickr terms. No images from either dataset are copied or trained.
- The custom `best.pt` and `last.pt` files currently have identical SHA-256 digests. Both, along with the original baseline, are preserved.

## License references

- Open Images download information: <https://storage.googleapis.com/openimages/web/download.html>
- COCO official Terms of Use: <https://cocodataset.org/dataset/termsofuse.htm>
- The Open Images per-image attribution information used for this selected subset is in `open_images_v7/manifests/attribution_records.json`.
- Visual Genome's existing import report documents why dataset-level repository metadata was not accepted as per-image image-use clearance.

## Models and compute

- Original baseline: repository-root `yolo11n.pt`, 5,613,764 bytes, SHA-256 `0ebbc80d4a7680d14987a577cd21342b65ecfd94632bd9a8da63ae6417644ee1`.
- Existing custom `best.pt` and `last.pt`: 21,240,931 bytes each; both SHA-256 `cdd694f376502f3eb5346fd8034c87b411367b279aad9b154cd26f2fe873a72f`.
- `.venv-1`: Python 3.11.9, Ultralytics 8.4.166, PyTorch 2.14.0+cpu; CUDA unavailable.
- At audit time: about 5.08 GiB available RAM and 105.34 GiB free on C:.
- The existing run recorded 5,075 seconds through epoch 19 for the same train/validation scale at batch 16. The new CPU run retains the historically used batch 16 and is estimated at about 89 minutes for 20 epochs (with early stopping); this is an estimate, not a guarantee. The previous 19-epoch run is direct evidence that batch 16 completed on this host. Storage is ample for a separate image copy, checkpoints, and evaluation outputs.

## Completed training and held-out evaluation

- Training completed all 20 epochs on CPU in 5,025.57 seconds (about 83.8 minutes), using all 420 train images. Validation used the separate 120-image split for early stopping; the 60-image test split was not used in training. The configured patience was 5 epochs.
- New best and last checkpoints are under `training/runs/yolo11n_byaku_openimages7_full600_20261011/weights/`. The original root `yolo11n.pt` and the pre-existing custom `best.pt`/`last.pt` hashes match their pre-run audit values.
- The held-out comparison used the same 60 images and 210 ground-truth boxes for both models. Custom vs. original baseline: precision 0.4385 vs. 0.3671, recall 0.3877 vs. 0.3934, mAP@0.5 0.3411 vs. 0.3235, and mAP@0.5:0.95 0.2364 vs. 0.2558.
- Model-only CPU inference was 66.80 ms/image (14.97 FPS) for the custom model and 66.96 ms/image (14.94 FPS) for the baseline. These figures exclude all BYAKU pipeline overhead and are not end-to-end real-time performance.
- Test support was person 33, bottle 38, book 71, chair 21, table 21, mobile_phone 13, laptop 13, and bag 0. Bag has no test ground truth and was not included as a model output class. Baseline `dining table` is only an approximate category counterpart to the target's generic `table`.
- The held-out comparison metrics and checkpoint hashes are summarized in this audit. Generated evaluation plots, copied test datasets, JSON/CSV output artifacts, and annotated false-positive/false-negative previews were removed during project cleanup; the training checkpoint and source dataset remain.

Repeat training with the prepared dataset and original baseline:

```powershell
& .\.venv-1\Scripts\python.exe -m Workflow.training.run_detector_training train
```

Repeat evaluation with a newly completed checkpoint (choose a unique evaluation name if providing one):

```powershell
& .\.venv-1\Scripts\python.exe -m Workflow.training.run_detector_training evaluate --custom-checkpoint Models\training_runs\yolo11n_byaku_openimages7_full600_20261011\weights\best.pt --output-root Models\training_runs\evaluations
```
