# Step 44 — COCO128 Source Inspection and Validation Gate

## Decision

**STOPPED before extraction to a dataset directory, filtered-label creation, and training.** The official COCO terms state that COCO does not own the image copyrights and that image use must abide by Flickr's terms. The downloaded COCO128 archive has no per-image license, Flickr source URL, photographer/creator credit, or attribution record. The archive therefore does not establish that every image in this subset is eligible for this intended reuse or preserve required per-image attribution. No image-level rights inference was made from the COCO annotation license.

This is not a claim that COCO is license-free. The official COCO terms and the checked archive distinguish the materials as follows:

- **COCO annotations and website:** Creative Commons Attribution 4.0 International; attribution/link and indication of changes are required under CC BY 4.0.
- **COCO images:** COCO Consortium expressly says it does not own image copyright; use must abide by [Flickr's terms](https://www.flickr.com/creativecommons/), and users accept full responsibility for dataset use and copies of copyrighted images. Individual image rights and credits are not present in this COCO128 package, so they could not be verified here.
- **Archive `LICENSE` and `README.txt`:** the README describes Ultralytics' included conversion/software material as GPL-3.0; that file is not a per-image license grant.

Official sources:

- [COCO Terms of Use](https://cocodataset.org/#termsofuse), specifically the rendered [Terms of Use content](https://cocodataset.org/dataset/termsofuse.htm)
- [Ultralytics COCO128 dataset instructions](https://docs.ultralytics.com/datasets/detect/coco128/)
- [Official Ultralytics COCO128 dataset YAML](https://raw.githubusercontent.com/ultralytics/ultralytics/main/ultralytics/cfg/datasets/coco128.yaml), which names the 128-image sample, class IDs, and the archive source
- [COCO annotations license](https://creativecommons.org/licenses/by/4.0/)
- [Flickr Creative Commons terms](https://www.flickr.com/creativecommons/)

The archive was downloaded from the official YAML's documented [Ultralytics release URL](https://github.com/ultralytics/assets/releases/download/v0.0.0/coco128.zip), not from an alternate dataset source. The COCO terms page initially returned only its SPA shell; the full official terms were inspected after the archive download. This ordering did not meet the requested "terms before download" sequence. Once the explicit image-rights caveat was found, no further image extraction or dataset use was performed.

## Download and archive inspection

- Archive: `Data/datasets/coco128/coco128.zip`
- Compressed size: **6,983,030 bytes**
- SHA-256: `61e5e3028863d8ffc3b81d6a514603954889f0edd5e4b44c4ce60b2da99aeb8e`
- ZIP CRC check: passed (`testzip()` returned no failing member)
- Archive entries: 128 JPEG image payloads and 128 YOLO label files, plus the GPL-3.0 README/license
- No full COCO 2017 archive or additional dataset was downloaded.

For an archive-level integrity check, image payloads were decoded in memory (not extracted or copied to a training tree): **128/128** decoded. The original images are not visually cleared/approved for training by this integrity check. Image payload files were not written out separately and no bounding-box overlay was generated.

## Annotation structure and checks

The official COCO128 class map contains 80 IDs (0–79). The ZIP holds the original YOLO normalized center-format labels. Across 128 label files:

| Check | Result |
|---|---:|
| Images / label files | 128 / 128 |
| Missing image-label pairs | 0 |
| Original annotation lines | 941 |
| Structurally valid source labels | 899 |
| Source labels rejected for boxes outside normalized image bounds | 42 |
| Other malformed label lines | 0 |
| Exact duplicate image payload groups (SHA-256) | 0 |

The 42 source lines were **not clipped or rewritten**. Because no authorized filtered dataset was created, these are counts from read-only validation of the archive's source labels. They do not describe a converted BYAKU dataset.

## Candidate target coverage in the source labels

Counts below are source labels that passed class/finite/range/box-bound checks. They are not a final BYAKU class map or imported annotations.

| BYAKU target | COCO source class | COCO ID | Valid source boxes | Images |
|---|---|---:|---:|---:|
| person | person | 0 | 244 | 58 |
| bottle | bottle | 39 | 18 | 6 |
| book | book | 73 | 29 | 6 |
| chair | chair | 56 | 33 | 9 |
| laptop | laptop | 63 | 3 | 2 |
| mobile_phone | cell phone | 67 | 8 | 5 |
| table | dining table | 60 | 11 | 9 |
| bag | no exact general `bag` class | — | 0 | 0 |

`cell phone` is the source label used for the mobile-phone concept. `dining table` is narrower than generic `table`; no broader remapping was finalized. COCO's `backpack` (ID 24; 6 valid boxes/4 images) and `handbag` (ID 26; 20/10) are distinct categories; neither was silently used as a generic bag or merged with the other. Laptop, phone, bottle, and book have especially low image counts for a held-out validation, so even with image rights cleared this sample would need to be represented as a smoke-test dataset, not evidence of generalization.

## Preparation status

- Eligible images with verified image-level terms/attribution: **0 established**
- Filtered YOLO images/labels, deterministic split, `data.yaml`, and class mapping: **not created**
- Visual previews and box alignment review: **not performed**
- Training and evaluation: **not permitted/performed**
- Existing `yolo11n.pt`, production pipeline, original BYAKU candidate dataset, and `UI/` were not modified.

Next step: establish the original Flickr license and required credit for every selected COCO image (using permitted public metadata only), then recreate the approved subset manifest and validate/split it. Do not train from this archive until that image-level review is complete.

## Tests

- Existing BYAKU dataset tests using `.venv-1`: **15 passed**.
- Full `Workflow/tests` suite using the same interpreter and offline model flags: **371 passed, 1 skipped**, with one existing Starlette/httpx `TestClient` deprecation warning.
- These are project regression tests, not tests of a created COCO128 training dataset; no filtered dataset was produced.
