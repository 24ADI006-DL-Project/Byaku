from __future__ import annotations

import json
import os
from pathlib import Path
from threading import Lock
from typing import Any


class FeedbackStore:
    SCHEMA_VERSION = 2

    def __init__(self, path: Path) -> None:
        self.path = path
        self._lock = Lock()

    def list(self, *, limit: int = 100) -> list[dict[str, Any]]:
        with self._lock:
            records = self._read()
        return list(reversed(records[-limit:]))

    def create(self, record: dict[str, Any]) -> dict[str, Any]:
        with self._lock:
            records = self._read()
            submission_id = record.get("submission_id")
            if submission_id:
                duplicate = next(
                    (item for item in records if item.get("submission_id") == submission_id),
                    None,
                )
                if duplicate is not None:
                    return duplicate
            records.append(record)
            self._write(records)
        return record

    def get_by_submission_id(self, submission_id: str) -> dict[str, Any] | None:
        with self._lock:
            return next(
                (record for record in self._read() if record.get("submission_id") == submission_id),
                None,
            )

    def find_exact_correction(
        self,
        *,
        predicted_class: str,
        visual_fingerprint: str,
    ) -> dict[str, Any] | None:
        if not visual_fingerprint:
            return None
        with self._lock:
            records = self._read()
            candidate = next(
                (
                    record for record in records
                    if record.get("correction_type") == "class_correction"
                    and record.get("review_status") == "corrected"
                    and record.get("original_predicted_class") == predicted_class
                    and record.get("visual_fingerprint") == visual_fingerprint
                    and not record.get("applied_to_future_inference", False)
                    and isinstance(record.get("human_corrected_name"), str)
                    and record["human_corrected_name"].strip()
                ),
                None,
            )
            return candidate

    def mark_applied_to_future_inference(
        self,
        correction_id: str,
        *,
        applied_at: str,
    ) -> dict[str, Any] | None:
        with self._lock:
            records = self._read()
            for record in records:
                if record.get("correction_id", record.get("feedback_id")) != correction_id:
                    continue
                if record.get("correction_type") != "class_correction" or record.get("review_status") != "corrected":
                    return None
                if not record.get("applied_to_future_inference", False):
                    record["applied_to_future_inference"] = True
                    record["applied_at"] = applied_at
                    self._write(records)
                return record
        return None

    def undo(self, feedback_id: str, *, undone_at: str) -> dict[str, Any] | None:
        with self._lock:
            records = self._read()
            for record in records:
                if record.get("feedback_id") == feedback_id:
                    if record.get("review_status") == "undone":
                        return record
                    record["review_status_before_undo"] = record["review_status"]
                    record["review_status"] = "undone"
                    record["undone_at"] = undone_at
                    self._write(records)
                    return record
        return None

    def _read(self) -> list[dict[str, Any]]:
        try:
            content = self.path.read_text(encoding="utf-8")
        except FileNotFoundError:
            return []
        records = json.loads(content)
        if isinstance(records, dict):
            if records.get("schema_version") != self.SCHEMA_VERSION:
                raise ValueError("feedback store has an unsupported schema version")
            records = records.get("records")
        if not isinstance(records, list) or any(not isinstance(item, dict) for item in records):
            raise ValueError("feedback store must contain a JSON list of records")
        return records

    def _write(self, records: list[dict[str, Any]]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = self.path.with_suffix(f"{self.path.suffix}.tmp")
        try:
            with temporary_path.open("w", encoding="utf-8") as output:
                json.dump(
                    {"schema_version": self.SCHEMA_VERSION, "records": records},
                    output,
                    ensure_ascii=False,
                    indent=2,
                )
                output.write("\n")
                output.flush()
                os.fsync(output.fileno())
            os.replace(temporary_path, self.path)
        finally:
            temporary_path.unlink(missing_ok=True)
