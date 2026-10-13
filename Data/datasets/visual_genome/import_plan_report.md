# Step 43 — Visual Genome Import Plan (Blocked)

## Decision

**STOPPED before image download.** The source metadata provides no image-level creator, license, copyright, or attribution fields for any record. The dataset repository's CC BY 4.0 tag is recorded, but is not treated as proof that each third-party image is individually cleared. In addition, many object-annotation URLs differ from the corresponding image-metadata URLs. No mismatch was silently resolved.

The eligibility gate therefore admitted **zero** images. The deterministic candidate ledger is [`import_plan.json`](./import_plan.json). It records source image IDs, image dimensions, both URLs, URL-match decision, rights state, exclusion reasons, and a pointer to the original source annotation row. Original names, synsets, object IDs and pixel coordinates remain unmodified in the referenced `source/annotations/objects.json.zip`.

## Verified counts

| Measure | Count |
|---|---:|
| Source image/annotation records scanned | 108,077 |
| Images with at least one structurally valid box mapped to the reviewed 74-class vocabulary | 88,494 |
| Such candidates with unresolved per-image rights/attribution | 88,494 |
| Such candidates with annotation/metadata URL mismatch | 83,139 |
| Such candidates with exact URL match but unresolved image rights | 5,355 |
| Source records without a valid mapped box | 19,583 |
| Mapped invalid boxes (object level) | 4,510 |
| Candidate images that also contain at least one invalid mapped box | 3,577 |
| Images independently cleared for this import | **0** |
| Planned image downloads | **0** |

URL mismatch and rights exclusion counts overlap: every mapped-box candidate lacks image-level clearance, while 83,139 also have a URL mismatch. Across the entire 108,077 records, 101,112 URL pairs mismatch and 6,965 match exactly. A matching URL does not resolve image rights.

## Class review and pilot policy

All 74 existing class IDs were reviewed. The mapping is unchanged: exact normalized labels only, no inferred synonyms or category merges. `import_plan.json` contains each class ID, source valid-box/image counts, and the pilot decision. Source counts are not counts of permitted, downloaded, or manually validated BYAKU training examples. Every class has zero eligible pilot images and zero imported boxes. Priority examples include `bag` (ID 2), `book` (8), `bottle` (10), `cell phone` (17), `chair` (18), `laptop` (38), `person` (44), `phone` (45), and `table` (62); none is cleared for this import.

The intended pilot cap is 100 images, but no pilot sample was selected because no record passed the rights gate. This plan is not a smoke-test dataset and is not a training or evaluation split.

## URLs, rights, space, and stop conditions

- Both annotation and image-metadata URLs are retained per mapped-box candidate. If they differ, the plan says `not_authorized_to_substitute_metadata_url`; no fallback download is made.
- Source repository license metadata: CC BY 4.0. Attribution details are present at the dataset level, but this is insufficient to establish image-specific permissions/credit for the third-party source photos.
- No image `HEAD` checks or downloads were initiated for this step. The prior two sample `HEAD` measurements are not representative enough for a subset size estimate.
- Authorized image bytes: **0**, because the eligible set is empty. No estimate is asserted for a hypothetical future subset.
- The current unversioned annotation archive and metadata remain the traceability source; image staging, conversion, hash deduplication and splitting were not attempted.

Next safe step: obtain a reliable, reviewable per-image rights/attribution basis and verify which URL identifies each image. Only then regenerate a capped deterministic plan and estimate space for the exact eligible sample.
