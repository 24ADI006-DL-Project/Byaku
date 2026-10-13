from __future__ import annotations

import argparse
import json
from pathlib import Path

from Workflow.tools.dataset_validation import validate_and_write_report


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate BYAKU YOLO dataset structure and labels.")
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "Data" / "datasets" / "byaku_detection",
    )
    args = parser.parse_args()
    result = validate_and_write_report(args.dataset_root)
    print(json.dumps(result["quality_report"], indent=2, ensure_ascii=False))
    return 0 if result["quality_report"]["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
