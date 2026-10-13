from __future__ import annotations

import argparse
import json
from pathlib import Path

from Workflow.tools.dataset_validation import prepare_dataset


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare the BYAKU detection dataset and annotation queue.")
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path(__file__).resolve().parents[2],
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "Data" / "datasets" / "byaku_detection",
    )
    args = parser.parse_args()
    result = prepare_dataset(args.repo_root, args.dataset_root)
    print(
        json.dumps(
            {
                "dataset_root": str(args.dataset_root.resolve()),
                "manual_annotation_queue_count": result["queue_count"],
                "split_counts": result["split_counts"],
                "validation": result["manifest"]["validation_result"],
                "ready_for_training": result["manifest"]["validation_result"]["ready_for_training"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
