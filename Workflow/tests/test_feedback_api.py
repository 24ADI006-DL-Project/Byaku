from fastapi.testclient import TestClient
import json
import base64

import Backend.fastapi_application as api
from Backend.feedback_store import FeedbackStore
import cv2
import numpy as np


def test_feedback_can_be_saved_loaded_undone_and_reloaded(monkeypatch, tmp_path) -> None:
    store_path = tmp_path / "review" / "history.json"
    monkeypatch.setattr(api, "_feedback_store", FeedbackStore(store_path))
    client = TestClient(api.app)
    payload = {
        "frame_id": "browser:session-1:23",
        "frame_number": 41,
        "track_id": 7,
        "original_prediction": {
            "class_name": "cup",
            "confidence": 0.91,
            "bounding_box": [10, 12, 80, 100],
            "horizontal_position": "left",
        },
        "corrected_value": "bottle",
        "correction_type": "class_correction",
        "review_status": "corrected",
    }

    saved_response = client.post("/api/feedback", json=payload)
    assert saved_response.status_code == 201
    saved = saved_response.json()
    assert saved["timestamp"]
    assert saved["frame_id"] == payload["frame_id"]
    assert saved["track_id"] == payload["track_id"]
    assert saved["original_prediction"] == payload["original_prediction"]
    assert saved["corrected_value"] == "bottle"
    assert saved["correction_type"] == "class_correction"
    assert saved["review_status"] == "corrected"

    monkeypatch.setattr(api, "_feedback_store", FeedbackStore(store_path))
    loaded_response = client.get("/api/feedback")
    assert loaded_response.status_code == 200
    assert loaded_response.json()["feedback"] == [saved]

    undone_response = client.post(f"/api/feedback/{saved['feedback_id']}/undo")
    assert undone_response.status_code == 200
    assert undone_response.json()["review_status"] == "undone"
    assert undone_response.json()["review_status_before_undo"] == "corrected"

    reloaded = client.get("/api/feedback").json()["feedback"]
    assert reloaded[0]["review_status"] == "undone"
    assert reloaded[0]["feedback_id"] == saved["feedback_id"]


def test_feedback_accept_and_skip_actions_are_saved_without_edits(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(api, "_feedback_store", FeedbackStore(tmp_path / "actions.json"))
    client = TestClient(api.app)
    original = {"class_name": "person", "confidence": 0.99, "bounding_box": [1, 2, 3, 4]}

    for action, status, corrected_value in (
        ("accept", "accepted", "person"),
        ("skip", "skipped", "skipped"),
        ("position_flag", "corrected", "flagged_for_position_review"),
        ("position_correction", "corrected", {"bounding_box": [2, 3, 9, 10]}),
    ):
        response = client.post(
            "/api/feedback",
            json={
                "frame_id": "frame-1",
                "track_id": 3,
                "original_prediction": original,
                "corrected_value": corrected_value,
                "correction_type": action,
                "review_status": status,
            },
        )
        assert response.status_code == 201
        assert response.json()["correction_type"] == action

    assert len(client.get("/api/feedback").json()["feedback"]) == 4


def test_feedback_rejects_unknown_correction_types_and_undo_missing_records(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(api, "_feedback_store", FeedbackStore(tmp_path / "history.json"))
    client = TestClient(api.app)

    invalid = client.post(
        "/api/feedback",
        json={
            "original_prediction": {"class_name": "chair"},
            "correction_type": "retrain",
            "review_status": "corrected",
        },
    )
    assert invalid.status_code == 422
    assert client.post("/api/feedback/missing/undo").status_code == 404


def test_confirmations_are_saved_per_detection_and_duplicate_submissions_are_idempotent(monkeypatch, tmp_path) -> None:
    store_path = tmp_path / "history.json"
    monkeypatch.setattr(api, "_feedback_store", FeedbackStore(store_path))
    client = TestClient(api.app)
    records = []
    for track_id, class_name in ((10, "bottle"), (11, "cup")):
        payload = {
            "frame_id": "frame-42",
            "frame_number": 42,
            "track_id": track_id,
            "submission_id": f"confirm-{track_id}",
            "original_prediction": {"class_name": class_name, "confidence": 0.9},
            "corrected_value": class_name,
            "correction_type": "confirm",
            "review_status": "confirmed",
        }
        response = client.post("/api/feedback", json=payload)
        assert response.status_code == 201
        records.append(response.json())
        assert client.post("/api/feedback", json=payload).json() == response.json()

    assert [item["track_id"] for item in client.get("/api/feedback").json()["feedback"]] == [11, 10]
    assert all(item["review_status"] == "confirmed" for item in records)
    assert json.loads(store_path.read_text(encoding="utf-8"))["schema_version"] == FeedbackStore.SCHEMA_VERSION


def test_empty_corrected_name_is_rejected_and_undo_preserves_record(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(api, "_feedback_store", FeedbackStore(tmp_path / "history.json"))
    client = TestClient(api.app)
    payload = {
        "submission_id": "empty-name",
        "original_prediction": {"class_name": "cup"},
        "corrected_value": "   ",
        "human_corrected_name": "   ",
        "correction_type": "class_correction",
        "review_status": "corrected",
    }
    assert client.post("/api/feedback", json=payload).status_code == 422

    payload.update({
        "submission_id": "valid-name",
        "corrected_value": "water bottle",
        "human_corrected_name": "water bottle",
    })
    saved = client.post("/api/feedback", json=payload).json()
    undone = client.post(f"/api/feedback/{saved['feedback_id']}/undo").json()
    assert undone["review_status"] == "undone"
    assert undone["review_status_before_undo"] == "corrected"
    assert client.get("/api/feedback").json()["feedback"][0]["human_corrected_name"] == "water bottle"


def test_legacy_history_is_preserved_when_first_versioned_record_is_written(tmp_path) -> None:
    store_path = tmp_path / "legacy.json"
    legacy_record = {"feedback_id": "legacy-1", "correction_type": "accept", "review_status": "accepted"}
    store_path.write_text(json.dumps([legacy_record]), encoding="utf-8")
    store = FeedbackStore(store_path)
    store.create({"feedback_id": "new-1", "submission_id": "new-submit"})

    persisted = json.loads(store_path.read_text(encoding="utf-8"))
    assert persisted["schema_version"] == FeedbackStore.SCHEMA_VERSION
    assert persisted["records"][0] == legacy_record
    assert persisted["records"][1]["feedback_id"] == "new-1"


def test_correction_memory_matches_only_the_same_visual_crop(monkeypatch, tmp_path) -> None:
    store = FeedbackStore(tmp_path / "memory.json")
    monkeypatch.setattr(api, "_feedback_store", store)
    frame = np.zeros((48, 64, 3), dtype=np.uint8)
    frame[5:35, 5:35] = 150
    success, encoded = cv2.imencode(".jpg", frame)
    assert success
    api._review_frame_cache.clear()
    api._review_frame_cache["review-token"] = (
        base64.b64encode(encoded.tobytes()).decode("ascii"),
        api.time.monotonic(),
    )
    client = TestClient(api.app)
    payload = {
        "frame_id": "frame-1",
        "review_frame_token": "review-token",
        "original_prediction": {"class_name": "cup", "confidence": 0.8, "bounding_box": [5, 5, 35, 35]},
        "human_corrected_name": "bottle",
        "corrected_value": "bottle",
        "correction_type": "class_correction",
        "review_status": "corrected",
        "submission_id": "crop-correction",
    }
    saved = client.post("/api/feedback", json=payload)
    assert saved.status_code == 201
    record = saved.json()
    assert record["image_dimensions"] == {"width": 64, "height": 48}
    assert record["visual_fingerprint"]
    assert record["applied_to_future_inference"] is False

    processed_frame = api._decode_base64_jpeg(base64.b64encode(encoded.tobytes()).decode("ascii"))
    matching = [{"class_name": "cup", "bounding_box": [5, 5, 35, 35]}]
    by_identity, pending = api._apply_remembered_corrections(processed_frame, matching, warnings=[])
    assert matching[0]["class_name"] == "bottle"
    assert matching[0]["model_class_name"] == "cup"
    assert matching[0]["remembered_correction"]["corrected_name"] == "bottle"
    assert matching[0]["remembered_correction"]["applied"] is False
    assert len(pending) == 1
    store.mark_applied_to_future_inference(
        record["correction_id"],
        applied_at="2026-01-01T00:00:00+00:00",
    )
    matching[0]["remembered_correction"]["applied"] = True
    objects = [{"track_id": None, "class_name": "bottle", "bounding_box": [5, 5, 35, 35]}]
    api._attach_remembered_correction_metadata(objects, by_identity)
    assert objects[0]["model_class_name"] == "cup"
    assert objects[0]["remembered_correction"]["applied"] is True
    assert matching[0]["remembered_correction"]["applied"] is True

    unrelated_frame = frame.copy()
    unrelated_frame[5:35, 5:35] = 20
    unrelated = [{"class_name": "cup", "bounding_box": [5, 5, 35, 35]}]
    api._apply_remembered_corrections(unrelated_frame, unrelated, warnings=[])
    assert "remembered_correction" not in unrelated[0]
    assert unrelated[0]["class_name"] == "cup"


def test_exact_correction_is_applied_to_only_one_matching_detection(monkeypatch, tmp_path) -> None:
    store = FeedbackStore(tmp_path / "memory.json")
    monkeypatch.setattr(api, "_feedback_store", store)
    frame = np.zeros((48, 64, 3), dtype=np.uint8)
    frame[5:35, 5:35] = 150
    fingerprint = api._visual_fingerprint(frame, [5, 5, 35, 35])
    store.create({
        "feedback_id": "correction-1",
        "correction_id": "correction-1",
        "correction_type": "class_correction",
        "review_status": "corrected",
        "original_predicted_class": "cup",
        "human_corrected_name": "bottle",
        "visual_fingerprint": fingerprint,
        "applied_to_future_inference": False,
    })
    detections = [
        {"track_id": 4, "class_name": "cup", "bounding_box": [5, 5, 35, 35]},
        {"track_id": 5, "class_name": "cup", "bounding_box": [5, 5, 35, 35]},
    ]

    _, pending = api._apply_remembered_corrections(frame, detections, warnings=[])

    assert [item["class_name"] for item in detections] == ["bottle", "cup"]
    assert len(pending) == 1


def test_corrected_value_is_persisted_as_human_name_when_name_field_omitted(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(api, "_feedback_store", FeedbackStore(tmp_path / "history.json"))
    response = TestClient(api.app).post(
        "/api/feedback",
        json={
            "original_prediction": {"class_name": "cup", "confidence": 0.8},
            "corrected_value": "water bottle",
            "correction_type": "class_correction",
            "review_status": "corrected",
        },
    )

    assert response.status_code == 201
    assert response.json()["human_corrected_name"] == "water bottle"
