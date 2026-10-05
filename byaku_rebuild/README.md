# BYAKU Backend Rebuild

This folder is the isolated workspace for an incremental rebuild of the BYAKU backend integration. Existing project files and the current AI pipeline remain the source of truth and must not be changed by this rebuild.

## Existing Project Map

- Frontend: `../web/public/index.html`, `../web/public/dashboard.html`, `../web/public/js/`, `../web/public/css/`, and `../web/public/assets/`.
- Node.js static server and HTTP/WebSocket relay: `../web/server/server.js`; package metadata and the `ws` dependency are in `../web/package.json`.
- Original project FastAPI API: `../api.py`.
- Rebuild FastAPI API and browser WebSocket protocol adapter: `api.py` (`/api/frame`, `/api/query`, `/api/status`, and `/ws`). Browser WebSocket camera messages own a session only; Python camera lifecycle routes remain a separate server-owned mode.
- Pipeline orchestration and model configuration: `../app.py` (`VisionAssistantApp`, `AppConfig`).
- YOLO detector: `../detection/detector.py` (`YOLODetector`, Ultralytics `YOLO`). The default local weights file is `../yolo11n.pt`.
- ByteTrack: `../tracking/tracker.py` (`ObjectTracker`, Ultralytics `BYTETracker`).
- Depth Anything V2: `../depth/depth_estimator.py` (`DepthEstimator`, Transformers image processor/model).
- Depth integration and spatial classification: `../spatial/depth_integration.py`, `../spatial/spatial_processor.py`, `../spatial/distance.py`, and `../spatial/position.py`.
- Scene graph and short-term memory: `../scene_graph/scene_graph.py`, `../scene_graph/relationships.py`, and `../scene_graph/memory.py`.
- Query parsing and spatial reasoning: `../query/query_processor.py` and `../query/spatial_query.py`.
- Narration and speech: `../narration/narrator.py`, `../speech/tts.py`, and `../speech/speech_input.py`.
- Existing tests: `../tests/`.

## Rebuild Plan

Proceed one step at a time. Add new backend-integration code and tests only inside this folder. Run a focused test for each step and record its pass/fail result before starting the next step.

1. Inspect and prepare this isolated workspace. Complete; see `IMPLEMENTATION_STATUS.md`.
2. Specify and test the backend message/API contract without connecting a camera or changing the existing UI.
3. Add a small rebuild-owned API boundary that delegates to the existing `VisionAssistantApp`; do not duplicate or replace AI modules.
4. Add and test real-input validation and backend lifecycle/status behavior.
5. Integrate one real frame through the existing detector, tracker, depth, spatial, scene-graph, query, narration, and speech components in separate validated increments.
6. Add and test the rebuild-owned WebSocket/relay integration while preserving existing message types.
7. Run end-to-end checks with real pipeline data and document environment limitations.

No mock assistant, fabricated AI result, substitute depth map, or duplicate AI module is part of this plan. Do not load models or run inference during Step 1.

## Current Browser Protocol Integration

The existing browser owns its camera through `getUserMedia` and sends the established `start_camera`, `stop_camera`, `frame`, `query`, and `ping` messages over the same-origin Node `/ws` relay. `byaku_rebuild/api.py` implements the FastAPI `/ws` adapter and maps browser frames and queries onto the existing frame/query functions. Browser camera messages manage a WebSocket session only; they never start the Python OpenCV `LiveCameraProducer`.

Frame responses echo the browser's numeric `frame_id` for correlation and include the monotonic server `frame_number`. When valid relative depth exists, `depth_preview` is a compact colorized JPEG generated from the actual depth map; the numerical depth used for reasoning remains unchanged. Query responses map real `speech_response.spoken_text` to `narration`; audio, events, and timing metrics are not produced. `GET /api/status` serves the dashboard's same-origin poll from cached committed state. WebSocket session and sequence state is process-local, inference is synchronous in a worker thread, and no queue or background worker is used.

`browser_protocol_adapter.py` adds only aliases backed by returned data for frontend compatibility: tracked detections become `tracks`, spatial edge fields receive the dashboard's expected names, and actual scene speech is exposed as `scene_narration`. The depth preview passes through when present. Unsupported dashboard fields remain absent. The original `web/` Node and browser sources are unchanged.

Step 21 verified the minimal live relay connection (`Client -> Node.js relay -> FastAPI backend`). Step 22 verified real YOLO11n object detection + ByteTrack and Depth Anything V2 inference over the Node.js relay. Step 23 validated the actual browser UI, static delivery, WebSocket streaming lifecycle, scene query grounding, and developer dashboard status mapping.

See `CONTRACT.md` for message schemas and error behavior, `INTEGRATION_PLAN.md` for module boundaries and risks, and `IMPLEMENTATION_STATUS.md` for verified test results.