# HTTP and browser WebSocket contract

Run the services with the command in the root [README](../README.md). The FastAPI backend listens on port `8765`; the Node relay in `UI/server/http_websocket_relay.js` serves the static frontend on port `3000` and forwards `/api/*`, `/health`, and `/ws` to the backend.

## HTTP

- `GET /health` returns the service name and `status: "running"`.
- `GET /api/status` reports backend connectivity, current camera connectivity, `last_frame_available`, the retained `last_frame`, and its in-memory `last_frame_preview` when available, the latest query/reasoning/narration, stage statuses, and timing values when available. A retained frame does not mean the camera is connected.
- `POST /api/frame` accepts a JPEG frame as base64 plus positive width/height and frame identity fields. Its result includes pipeline status and server-side timing. `initialization_ms` is one-time adapter/model setup; `processing_ms` is the current frame's processing time; `timing_source` identifies the clock (`server_perf_counter`).
- `POST /api/query` accepts `{"query":"..."}` and answers from the latest committed scene. Reasoning status is set by query execution, not by receiving a frame.
- `POST /api/camera/start`, `/api/camera/capture`, and `/api/camera/stop` control the optional Python camera producer. Browser camera access remains owned by the browser.

Invalid requests use HTTP validation/error responses; overlapping synchronous processing or camera operations can return HTTP 409.

## Browser WebSocket

Connect through the frontend proxy at `ws://127.0.0.1:3000/ws`. The server sends an initial `status` message. Supported client message types include:

- `{"type":"ping","ts":42}` — returns `pong`.
- `{"type":"start_camera"}` / `{"type":"stop_camera"}` — marks the browser capture session; it does not open or release a server camera.
- `{"type":"frame","frame_id":1,"width":640,"height":480,"data":"<base64 JPEG>"}` — submits one frame for synchronous processing.
- `{"type":"query","text":"What is to the left of the bed?"}` — runs scene reasoning.

Frame and query messages return their actual result payloads; invalid messages and processing failures return structured `error` messages. No browser speech-synthesis capability or playback acknowledgement is currently sent to the backend. Narration text therefore does not mean that audio was played; server-side TTS is reported unavailable.

## Feedback

- `GET /api/feedback?limit=100` returns locally persisted review actions, newest first (limit 1–500).
- `POST /api/feedback` stores the frame reference, tracking ID, timestamp, original prediction, corrected value, correction type, and review status.
- `POST /api/feedback/{feedback_id}/undo` marks the action as undone without deleting its audit-history record.
- Exact normalized-crop matches may appear on live detections as `model_class_name` (the unchanged model prediction) and `remembered_correction`; such a correction is applied to that one inference only.
- The workflow dashboard's Human Review panel shows the latest processed camera frame with detections and allows per-object confirmation or class correction, with persistent history and audit-preserving undo.
- Feedback is stored in `Data/feedback/review_history.json`. Frame images are not written to that history; recent frames are cached in backend memory for exact-crop fingerprinting, and tokens older than 180 seconds are ignored.
- Feedback collection does not retrain or modify model weights. No training/export workflow consumes these records.
