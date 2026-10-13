# BYAKU Pre-Training Quality Gate Report (Phase 4)

## Quality Gate Status: PASSED

- **Total Valid Images**: 600 (Train: 420, Val: 120, Test: 60)
- **Total Valid Bounding Boxes**: 2167
- **Zero Split Leakage**: Verified 0 overlapping images across splits.
- **Exact Duplicate Hashes**: Verified 0 duplicate images.
- **Class Mappings**: Verified exact 8 target class IDs (0: person to 7: bag).
- **Coordinate Normalization**: 100% of bounding boxes normalized in [0.0, 1.0].
- **Attribution & Rights**: 100% of images verified with Creative Commons licenses and author credits recorded in `attribution_records.json`.
- **Note on `bag`**: Open Images V7 contains 0 bounding boxes for class MID `/m/0n5v01m`. Documented as expected; no synthetic boxes invented.
