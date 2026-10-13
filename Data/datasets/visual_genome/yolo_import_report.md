# Step 43 — Visual Genome YOLO Import Report

## Outcome

**Blocked before download and conversion.** No Visual Genome image was downloaded, no YOLO label was emitted, no train/validation/test split was created, and no image overlay could be visually inspected. This follows the required stop condition: the available files do not establish image-level reuse rights and attribution for any image. Dataset-level CC BY 4.0 metadata alone was not treated as per-image clearance.

The deterministic pre-download plan is [`import_plan.json`](./import_plan.json), with a human-readable summary in [`import_plan_report.md`](./import_plan_report.md). Earlier Step 41 integration and class-analysis reports were one-time investigations and are not part of the current supported import workflow; upstream license and URL uncertainty are documented here.

## 1. Images and exclusion counts

| Measure | Count |
|---|---:|
| Source records inspected | 108,077 |
| Potential candidate images with at least one valid box mapped to the reviewed class set | 88,494 |
| Eligible for download | **0** |
| Downloaded | **0** |
| Corrupt/truncated download files | **0** (no download attempted) |
| Excluded mapped-box candidate images | 88,494 |
| Source records without a valid mapped class box | 19,583 |

Exclusions among the 88,494 candidate images:

- **88,494**: image-level rights/creator/attribution not independently verified.
- **83,139**: annotation URL differs from image-metadata URL; this overlaps with the rights exclusion and was not silently substituted.
- **5,355**: URLs match exactly, but per-image rights remain unresolved, so these are also excluded.
- **4,510** mapped object boxes have structurally invalid coordinates; 3,577 candidate images also contain at least one invalid mapped box. Invalid boxes were not clipped. No YOLO conversion was performed.

Across all source records, 101,112 annotation/metadata URL pairs differ and 6,965 match. Neither URL matching nor URL accessibility establishes image-level reuse rights. Per-record IDs, dimensions, both URLs, the decision and exclusion reason are in the plan. Source object names, synsets, IDs and original pixel `x,y,w,h` values remain traceable in the referenced annotation ZIP.

## 2. Classes and IDs

The source class map remains the reviewed 74-class map in [`class_mapping.json`](./class_mapping.json), with IDs unchanged. No synonyms, plurals, related classes or broad/narrow categories were merged. No separate final YOLO class mapping was created because **no classes have eligible images**.

Priority source classes include bag (2), book (8), bottle (10), cell phone (17), chair (18), laptop (38), person (44), phone (45), and table (62). Their source annotation counts are not imported ground truth. Every class has **0 eligible images and 0 imported boxes**; per-class source coverage and decisions are included in `import_plan.json` and [`class_coverage_analysis.json`](./class_coverage_analysis.json). Sparse labels remain deferred according to the reviewed Step 41 map.

## 3. YOLO dataset and split validation

| Split | Images | Annotations/boxes |
|---|---:|---:|
| Train | 0 | 0 |
| Validation | 0 | 0 |
| Test | 0 | 0 |

No `yolo/` dataset, `data.yaml`, image/label pairs or label files were created. Therefore there are no YOLO coordinate ranges or image/label pairings to validate. There is no meaningful split or test set; the planned maximum-100-image sample did not proceed and is not a smoke-test dataset.

## 4. Duplicates and visual QA

- Duplicate-by-content checks: **not run**, because no eligible images were downloaded. Duplicate prevention must be run before any future split.
- Split leakage: **no split was created**; no leakage-free split is claimed.
- Visual QA: **not performed**, because no images could be imported. Source coordinates were inspected structurally in the annotation records, not visually confirmed against downloaded pixels.
- Box convention in source records is pixel `x,y,w,h`, as reflected in the source schema and earlier Step 41 inspection. The importer conversion utility can normalize valid boxes, but Step 43 conversion was not run without eligible images.

## 5. Disk usage

- Image bytes downloaded/staged: **0**.
- No YOLO image or label bytes were created.
- `import_plan.json`: 92,838,794 bytes, recording 88,494 per-image candidate exclusions and source references.
- Existing annotation ZIPs: 57,104,783 bytes combined.
- Free C: space at inspection: 115,170,177,024 bytes.
- A useful estimate for a future cleared image sample cannot be made from the two earlier individual-image size probes; they are not representative.

## 6. Tests

Using `.venv-1\Scripts\python.exe` (Python 3.11.9), with `HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1`:

- Focused Visual Genome importer plus existing BYAKU dataset tests: **38 passed**.
- Full `Workflow/tests` suite: **371 passed, 1 skipped**, one existing Starlette/httpx `TestClient` deprecation warning.
- No dependency was installed. The earlier Transformers/regex environment issue did not recur.

These tests validate importer safeguards, including stable IDs, ambiguous labels, box validation, duplicate groups, splits, URL mismatch handling, and rejection of missing or mismatched image-rights evidence. They do not imply that the Visual Genome images were licensed, imported, or visually validated.

## 7. Readiness and unresolved items

**Not ready for training or evaluation.** This is a blocked import plan, not a dataset pilot: zero images and zero labels were produced.

All candidate images remain excluded pending independently verified image-level rights/attribution. URL authority also remains unresolved for mismatched rows; no substitution or authentication bypass was attempted. Accessibility/hazard labels and other sparse classes retain their prior reviewed status; none were brought into a final YOLO class list.

The importer now requires image-specific rights evidence matching the exact image ID and URL, including an evidence reference, license, attribution, reviewer, and review date, before staging. This gate supplements—but cannot replace—a real rights review.

## 8. Production/training safety

No training, fine-tuning, export, or model replacement was run. Existing BYAKU candidate images/manifests, production detector, UI, backend inference pipeline, and `UI/` were not modified.

**TRAINING NOT STARTED — awaiting explicit approval.**
