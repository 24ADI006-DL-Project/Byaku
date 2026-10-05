# BYAKU Integration Plan

## Scope and Evidence

This plan records verified module contracts and implemented integration boundaries. The original contract tests use synthetic detections and relative-depth arrays; the live API and camera producer now connect the existing modules synchronously.

The module contracts below were checked against their current source. Each processing module has an independent synthetic test file; the API has its own FastAPI contract tests. The five tests in `tests/test_integration_contracts.py` validate a synthetic tracked-object path and known cross-module boundaries without model inference.

## Verified Module Contracts

### `spatial_relationships.py`

- Public functions: `bounding_box_iou(first_box, second_box)`, `image_shape_from_frame_size(frame_size)`, and `calculate_spatial_relationships(detections, image_shape, depth_map=None, *, frame_id=None, center_zone_fraction=0.2, overlap_threshold=0.1, depth_difference_threshold=0.05, higher_depth_is_nearer=True)`.
- Detection input: iterable of mappings. A box is required under `bounding_box` or `bbox` and uses pixel `(x1, y1, x2, y2)` coordinates. `class_name` is preferred; `class`, `name`, and `object` are fallbacks. `track_id` or `id` and `confidence` are optional. A supplied finite `confidence` is preserved as a float; missing confidence remains absent.
- Internal `image_shape` is canonically `(height, width)`. The explicit `image_shape_from_frame_size({"width": w, "height": h})` adapter validates positive integer fields and returns `(h, w)`. A supplied depth map must be 2D and exactly match `(height, width)`; mismatched maps are ignored with a warning rather than resized. Relative depth is the median finite value in each clipped box ROI.
- Output: `frame_id`; `image_shape`; `objects` with `object_id`, `track_id`, `class_name`, clipped float `bounding_box`, pixel `center`, positions, optional `relative_depth_value`, and optional `confidence`; `relationships` with `subject`, `relation`, `object`, and optional `iou` or `depth_difference`; `warnings`.
- Track IDs become `track:<value>` object IDs. Untracked IDs use `temporary:<frame-id>:<detection-index>`. If omitted, spatial processing assigns a unique `auto-<n>` frame ID; integration should pass one shared unique frame ID to spatial calculation and SceneMemory for each processed frame. Directional geometry is emitted reciprocally. Higher depth values mean nearer by default; the polarity is still not carried in the output.
- Invalid thresholds raise `ValueError`; malformed boxes and invalid dimensions/depth produce skipped records or warnings. Uses NumPy. Independently covered by `tests/test_spatial_relationships.py`.

### `scene_graph.py`

- Public functions: `build_scene_graph(spatial_output)`, `get_objects(graph)`, `find_relationships_for_object(graph, object_id)`, `find_objects_by_class(graph, class_name)`, and `query_relationship(graph, source, destination, relationship_type=None)`.
- Input is the spatial output mapping above. The builder returns an `nx.MultiDiGraph`; nodes are keyed by `object_id` and preserve input object attributes including confidence. Edge keys and `relation` attributes contain relationship types; evidence such as `iou` and `depth_difference` stays on the edge. Graph-level `warnings`, `image_shape`, and `frame_id` are stored. Legacy `frame:<index>` IDs are normalized to frame-scoped temporary IDs and relationship endpoints are aliased.
- Duplicate node IDs keep the first node. Duplicate `(source, destination, relation)` edges, invalid relationship types, and references to missing/self nodes are skipped with warnings. `similar_depth` is retained.
- Depends on installed NetworkX. Independently covered by `tests/test_scene_graph.py`.

### `scene_memory.py`

- Public API: `SceneMemory(stale_after_frames=5, history_size=20, stale_retention_frames=300)`; `update(frame_data, *, frame_number=None, timestamp=None)`; `get_visible_objects()`, `get_temporarily_missing_objects()`, `get_stale_objects()`, `get_object(tracking_id)`, and `get_recent_observations(tracking_id, limit=None)`.
- `update` accepts a spatial output mapping (reads its `objects` list), one object mapping, an iterable of object mappings, or `None`. Object records require `class_name` and `bounding_box`/`bbox`; center, `relative_depth_value`, and `confidence` are optional. Tracked records have canonical IDs `track:<value>`. Spatial `temporary:<frame-id>:<index>` IDs are preserved for that frame. Legacy `frame:<index>` inputs are upgraded to `temporary:<memory-frame>:<index>`; raw records without IDs get temporary IDs from the memory frame.
- Update output contains `frame_number`, `visible_objects`, `temporarily_missing_objects`, `stale_objects`, and `warnings`. Records contain class, box, center, optional depth/confidence, `last_seen_frame`, `last_seen_timestamp`, status, and bounded observations. Missing becomes stale when age reaches `stale_after_frames`. Stale records are evicted only when age since last seen is greater than `stale_retention_frames` (default 300); re-observation before eviction restores visibility and history. Temporary IDs are not retained across frames.
- Invalid configuration raises `ValueError`; invalid records and repeated/non-increasing frame numbers are handled with warnings. Independently covered by `tests/test_scene_memory.py`.

### `scene_description.py`

- Public function: `describe_scene(spatial_output=None, *, scene_graph=None, scene_memory=None)`.
- It accepts spatial output, an `nx.MultiDiGraph`, and a `SceneMemory` instance or update snapshot. When memory is supplied, its visible-object set is authoritative. It returns `description`, `visible_object_count`, `described_relationships`, and `warnings`.
- It only verbalizes supplied, validated relation records. Directional geometry is checked against current centers/boxes; overlap needs both positive supplied IoU and actual box intersection; relative-depth language needs finite values for both current objects. It collapses reciprocal/symmetric duplicates.
- Uses NetworkX and the spatial IoU helper. Independently covered by `tests/test_scene_description.py`.

### `query_reasoning.py`

- Public function: `answer_query(query, *, spatial_output=None, scene_graph=None, scene_memory=None)`.
- It consumes a query string and the same spatial/graph/memory inputs as `describe_scene`. It calls `describe_scene` to obtain the validated current relationships. Output fields are `query`, `answer`, `status`, `matched_objects`, `evidence`, and `warnings`.
- Current statuses include `visible`, `missing`, `stale`, `not_found`, `ambiguous`, `insufficient_evidence`, `unsupported`, and `empty_scene`. Parsing is deterministic and limited to its implemented question patterns. Inverse directional records can answer the equivalent reverse query; evidence retains the source relationship.
- Depends on NetworkX and `scene_description.py`. Independently covered by `tests/test_query_reasoning.py`.

### `speech_response.py`

- Public functions: `prepare_spoken_response(query_result)` and `prepare_spoken_scene_summary(scene_result)`.
- The first consumes the query-result mapping; the second consumes the scene-description mapping. Both return `spoken_text`, `source_status`, and `warnings`. Warnings remain structured rather than being appended to speech; exact duplicate sentences are removed and speech is capped at two sentences.
- It is text formatting only and imports no audio, microphone, or TTS runtime. Independently covered by `tests/test_speech_response.py`.

### `api.py`

- Public FastAPI application `app`; implemented endpoints include `GET /health`, `GET /`, `POST /api/frame`, `POST /api/query`, and the camera lifecycle endpoints `POST /api/camera/start`, `/api/camera/capture`, and `/api/camera/stop`.
- `POST /api/frame` validates the supplied JPEG request and passes the image through the existing detector, depth adapter, and orchestrator. `POST /api/query` queries the latest committed scene.
- Camera start configures one process-local `LiveCameraProducer`; capture synchronously requests one frame and returns the resulting `frame_result`; stop and app shutdown release the camera. A nonblocking process-local operation guard rejects overlapping camera lifecycle/capture operations. It does not coordinate multiple workers.
- Camera errors use the common structured error shape with `stage: "camera"`. The API and camera integration are covered by `tests/test_api.py` and `tests/test_live_camera_api.py` using fake processing components and capture devices.
- `GET /api/status` exposes the latest cached committed frame/query and observed stage/session status without model initialization or fabricated metrics.
- `WebSocket /ws` preserves the browser message protocol relayed unchanged by `web/server/server.js`. Browser start/stop messages control only a WebSocket session; browser JPEG frames are validated and adapted to `FrameRequest`, and queries map to `PipelineOrchestrator.query()` through the existing API helper. Browser frame IDs are echoed while internal IDs and server frame numbers remain unique/monotonic.
- `browser_protocol_adapter.py` is a pure compatibility layer for actual Step 19 results. It maps tracked detections to dashboard `tracks`, relationship keys to the dashboard's `source`/`target`/`relationship` names, scene summary speech to `scene_narration`, matched query objects to `reasoning.objects`, and speech text to `narration`. Unsupported values remain omitted; it creates no detections, distance categories, events, timing, preview, or audio data.
- WebSocket inference is dispatched through Starlette's worker thread path. A shared nonblocking processing lock protects REST and WebSocket frame/query work. Busy requests fail immediately; no work is queued. Browser messages never start or stop the Python `LiveCameraProducer`.
- Focused protocol coverage lives in `tests/test_websocket_protocol.py`; the REST API suite also tests `/api/status`. These tests use synthetic JPEGs and fake detector/depth adapters.

## Proposed Execution Paths

Do not run the entire chain serially for every user interaction.

1. **For each valid frame:** acquire one frame and its monotonically increasing frame number/timestamp; run detector/tracker and depth on that same frame; adapt those outputs to the spatial input; call `calculate_spatial_relationships`; update `SceneMemory` exactly once for that valid processed frame. Build a per-frame scene graph if another same-frame consumer needs it. A graph can also be built on demand from the latest spatial result.
2. **For a scene summary:** on request, call `describe_scene` with the latest spatial result, matching graph snapshot (or no graph), and the memory instance; then call `prepare_spoken_scene_summary`.
3. **For a query:** call `answer_query(text, spatial_output=latest_spatial, scene_graph=latest_graph, scene_memory=memory)` so it can use current-frame geometry and relationships while memory decides whether objects are still visible. Then pass that result to `prepare_spoken_response`.
4. **For browser sessions:** same-origin `/ws` is relayed unchanged by Node to FastAPI. `start_camera`/`stop_camera` mark only the browser session. The browser remains the sole camera owner; each JPEG frame is adapted to the existing `FrameRequest` contract and processed through the existing API/pipeline. Query text is sent to the existing query processor and its spoken text is mapped to `narration`.
5. **For API camera capture:** `POST /api/camera/start` opens the configured Python producer; each `POST /api/camera/capture` processes one frame through the same contract; `POST /api/camera/stop` releases it. This is a separate server-owned camera mode and must not be invoked by the browser WebSocket protocol.
6. **For dashboard state:** `GET /api/status` returns cached committed results and only observed stage/session states; unsupported timings, events, audio, and previews remain unavailable/omitted.

Only valid frames should update memory. A capture failure or inference failure must not be passed as an empty object list: `SceneMemory.update([])` advances the frame and marks prior objects missing/stale. A valid frame with zero detections is different and should update memory with an empty spatial result.

## Proposed Boundary Schemas

These are integration-adapter schemas, not new current module APIs.

### Detector/tracker and depth to spatial calculation

```python
detections = [
    {
        "track_id": 7,                 # int or None
        "class_name": "bottle",       # str
        "confidence": 0.90,            # float; currently discarded by spatial calculation
        "bounding_box": [2.0, 2.0, 12.0, 12.0],  # pixel xyxy
    }
]
frame_id = 42                            # unique for this processed frame
image_shape = image_shape_from_frame_size({"width": 40, "height": 20})  # -> (20, 40)
depth_map = ndarray_or_none               # 2D float, same frame's HxW, relative only
spatial = calculate_spatial_relationships(detections, image_shape, depth_map, frame_id=frame_id)
```

The existing image-detection script prints `class_name`, confidence, and `bounding_box`, without a tracking ID. The tracking test prints class, integer ID, confidence, and `bbox`; it does not expose a reusable normalized records API. A future adapter must explicitly combine/map those outputs instead of assuming they already match. Spatial processing now carries the optional `confidence` field when supplied.

### Spatial calculation to graph and memory

Pass the exact spatial output mapping to both `build_scene_graph(spatial)` and `SceneMemory.update(spatial, frame_number=n, timestamp=t)`. For integer tracker IDs, outputs align as `track:7` in spatial objects, graph nodes, and memory records. For untracked observations, pass a unique `frame_id` and matching `frame_number`; spatial, graph, and memory then share `temporary:<frame-id>:<index>`. The spatial output is a per-frame snapshot; the graph is a directed per-frame snapshot; memory is the temporal state.

The former untracked-ID mismatch is fixed and covered by regression tests. A caller-supplied frame ID must be unique per frame; spatial processing generates unique `auto-<n>` IDs when it is omitted, but an orchestrator should pass the same explicit unique frame ID to all consumers. SceneMemory remaps a reused temporary ID rather than making it persistent; if a caller reuses frame IDs, graph/memory snapshots may no longer align and a warning is emitted.

### Graph/memory to description and query

```python
description = describe_scene(spatial, scene_graph=graph, scene_memory=memory)
query_result = answer_query(
    "What is to the left of the chair?",
    spatial_output=spatial,
    scene_graph=graph,
    scene_memory=memory,
)
```

Use a graph built from the same spatial frame as the spatial output and memory update. Pass current spatial data as well as memory for current location labels; memory stores center/box/depth/history but does not retain horizontal/vertical position labels or image dimensions. `SceneMemory` is authoritative for visibility, while query/description use spatial/graph relationships filtered to currently visible IDs.

### Query result to speech and WebSocket API

`answer_query` returns `{query, answer, status, matched_objects, evidence, warnings}`. `prepare_spoken_response(query_result)` returns `{spoken_text, source_status, warnings}`. The WebSocket adapter maps browser `text` to the query helper and aliases the actual `spoken_text` as `narration`; it does not produce an `audio` field. The browser can use its existing speech synthesis for that narration.

## Synthetic Tracked-Object Example

This example is exercised by `test_tracked_frame_contract_flows_through_query_and_speech`; values are synthetic, not model results. The input has two detections (bottle ID 7 at `[2,2,12,12]`; chair ID 9 at `[25,2,35,12]`), `image_shape=(20,40)`, and a same-sized synthetic depth map with higher values in the bottle region.

1. Spatial output contains bottle object ID `track:7`, center `(7,7)`, horizontal position `left`, a finite relative depth value, and a `left_of` relationship to `track:9` (plus its reciprocal and relative-depth edges).
2. `build_scene_graph(spatial)` uses `track:7` and `track:9` as node IDs and preserves relation types/evidence on directed edges.
3. `memory.update(spatial, frame_number=1, timestamp=1.0)` yields a currently visible `track:7` record with last-seen frame 1. These tracked IDs match the graph and spatial result.
4. `describe_scene(spatial, scene_graph=graph, scene_memory=memory)` describes the supplied left/depth relationships.
5. `answer_query("What is to the left of the chair?", ...)` returns a visible bottle result and relationship evidence.
6. `prepare_spoken_response(query_result)` produces concise text from that answer; it does not generate audio.

## Confirmed Risks and Open Questions

- **Temporary ID mismatch (fixed):** spatial processing, graph construction, and memory now share `temporary:<frame-id>:<index>` for a frame. Frame-ID uniqueness remains an orchestrator responsibility; repeated IDs are guarded against by memory remapping and warnings.
- **Confidence loss (fixed):** spatial processing preserves optional finite `confidence`; SceneMemory and query matched-object projection also retain it. Sources without confidence continue to omit the field.
- **Coordinate conventions (fixed):** internal image shape remains `(height,width)` and `image_shape_from_frame_size({"width": w, "height": h})` converts the API frame dimensions before spatial processing.
- **Depth contract (confirmed):** depth must be a same-frame 2D map with exact image dimensions or it is ignored. Higher values mean nearer only under the spatial function's default; the choice is not recorded in its output. Do not pair depth from one captured frame with boxes from another, and do not report meters.
- **Depth relation validation (confirmed):** `scene_description` checks that both current objects have finite depth values but does not verify the sign of `depth_difference` against a `nearer_than`/`farther_than` label. Build relationships and depth values from the same frame and same polarity setting; stale graphs need not be reused.
- **Memory retention (fixed):** stale records are evicted after their age exceeds `stale_retention_frames` (default 300 frames); history remains bounded per object. Reappearance before eviction updates the existing tracked entry. The default now changes former indefinite stale retention and is configurable.
- **Frame lifecycle (confirmed):** memory needs monotonically increasing frame numbers for expected age semantics. Calling `update([])` on a failed frame incorrectly advances missing/stale state; distinguish valid empty detections from missing frame/inference failures.
- **Blocking and throughput (measured):** the optimized CPU webcam depth test measured 1.56 FPS at reduced input. WebSocket inference runs in a worker thread and has no queue; the browser's five-second timeout can expire, and frames are skipped while one is outstanding. Full browser-through-Node behavior and end-to-end throughput remain unverified.
- **Relationship duplication (confirmed):** spatial calculation emits reciprocal direction edges; the graph preserves them. Description/query currently collapse or interpret reciprocal edges. Keep one same-frame graph snapshot and avoid separately concatenating graph and spatial edge lists.
- **Exception boundaries (implemented):** frame-processing failures return structured REST or WebSocket errors, and transactional orchestration preserves the last committed scene. Protocol errors are scoped to the affected WebSocket session.
- **API readiness (implemented):** REST frame/query/camera routes, `/api/status`, and the browser WebSocket protocol adapter reuse the existing adapters and orchestrator. Failed processing does not commit a new scene.
- **Original frontend compatibility (adapter-only):** actual result aliases are covered by synthetic tests, but original browser/Node sources remain unchanged. No live browser-to-Node-to-FastAPI end-to-end run was performed. Dashboard fields without backend sources intentionally remain unavailable.
- **Concurrency and freshness (limited):** a process-local fail-fast guard prevents overlapping REST/WebSocket frame and query processing; it does not coordinate multiple workers. Session/status state is also process-local.
- **Blocking and shutdown (confirmed):** model inference is synchronous within a worker thread. The browser's existing five-second frame timeout can expire before inference completes; busy frames are rejected rather than queued. WebSocket disconnect only clears session state. The Python producer is released by its explicit stop/app lifespan, not by browser session messages.
- **Relay/startup and tracker state (known):** the existing Node relay buffers at most two outbound messages while connecting, and its WebSocket payload limit is 6 MB. The browser's frames are JPEGs and no queue is introduced. Persistent third-party ByteTrack temporal state cannot be rolled back after downstream failure.

## Step 10 Validation

At Step 10, `tests/test_integration_contracts.py` checked tracked/untracked ID agreement and a synthetic route through spatial calculation, graph, memory, description, query, and speech. Current REST/WebSocket API behavior is covered by the focused API and protocol tests described above. These tests do not establish real model accuracy, multi-worker coordination, browser rendering, or end-to-end responsiveness.

## Step 11 Contract Fixes

The temporary-ID, optional-confidence, dimensions-conversion, and stale-retention contracts above were updated and covered by regression tests. The complete suite passed after these changes. No API endpoint or inference pipeline was added.

## Step 21 Minimal Live Integration Verification

Step 21 verified the minimal live connection between the Node.js relay, FastAPI backend, and simulated browser client:

- **FastAPI backend (Test A):** `/health`, `/api/status`, and WebSocket `/ws` connection, initial status reporting, ping/pong echo, and session cleanup on disconnect verified.
- **Relay path (Test B):** Verified via a Python relay harness (simulating Node forwarding) and confirmed with the live Node.js relay process using `run_node_relay_test.py`. Real path `WS client -> Node relay (3001) -> FastAPI (8766)` confirmed working for ping/pong routing. The browser `start_camera` message does not start the Python camera producer.
- **Frame and query payload (Test C):** Synthetic JPEG frame via WebSocket returned `frame_result` with integer `frame_id` echoed and valid dimensions; subsequent query returned `query_result` with valid status and omitted unsupported fields (no audio, no depth preview, no fabricated timing).
- **Remaining gaps & boundaries:**
  - Real browser UI execution in a live browser engine remains unverified (no live browser session test).
  - Node.js `/api/*` HTTP body cap is 1 MB, whereas the WebSocket payload limit is 6 MB.
  - Model inference in tests used mock adapters; full live inference with YOLO and Depth Anything weights was not part of this minimal integration check.

## Step 22 Real AI Inference Integration

Step 22 verified the integration of real YOLO11n object detection and Depth Anything V2 Small inference with the rebuild pipeline, direct API, and live Node.js relay:

- **Single-Image Inference:** Loaded local `yolo11n.pt` and cached `Depth-Anything-V2-Small-hf` on `test1.jpg` (1104×736). Detected 7 real objects (chair, bed, laptop, potted plant, book, bottle); produced full-resolution relative-depth map (higher values nearer, no metric distance). Full pipeline computed spatial relationships and scene description without fabrication. Visual artifacts saved under `byaku_rebuild/results/`.
- **Direct API (`/ws`):** Tested with real models and real JPEG frames via FastAPI WebSocket client. Validated `frame_result` structure, stage status (`ready`), real tracks, echoed browser frame ID, and valid query answering ("What is to the left of the bed?").
- **Live Node Relay E2E:** Live Node.js relay process (port 3001) forwarding to FastAPI (port 8766). Submitted multi-frame sequence (`test1.jpg` frame 201, `test2.jpg` frame 202). Confirmed track continuity and memory tracking across frames, sub-3s latency per frame, and successful spatial reasoning over the relay.
- **Boundaries & Observations:**
  - Inference runs synchronously on CPU inside Starlette threadpool; frame latency is ~1.7–2.6s.
  - Live browser rendering (DOM, speech synthesis, camera capture via WebRTC/canvas) remains to be verified in a real browser session.

## Step 23 Actual Browser UI Validation

Step 23 validated the actual browser UI, static asset delivery, and Node.js relay communication with the rebuilt FastAPI backend:

- **Static Delivery & Proxies:** Verified that the Node.js server (`server.js`) delivers all HTML, CSS, JavaScript, and image assets cleanly with valid MIME types and proxies `/api/*` and `/ws` without payload corruption.
- **Workflow & UI Contract:**
  - `index.html`: Verified the intro screen (`#intro-screen`), skip action (`#intro-skip`), assistant panel state transitions (`Idle` -> `Listening` -> `Analyzing` -> `Speaking`), orb button (`#start-listening`), and camera view card.
  - `dashboard.html`: Verified pipeline track module status mapping (Camera, YOLO, ByteTrack, Depth Anything V2, Spatial, Scene Graph, Reasoning, Narration, TTS), real detection rows (`#detection-rows`), spatial positions (`#spatial-rows`), scene graph node/edge summaries, and query reasoning answers.
- **Camera Workflow & Permission Fallback:**
  - Verified browser camera integration via `navigator.mediaDevices.getUserMedia`.
  - Audited error boundaries in `camera.js` for permission denial (`NotAllowedError`), missing devices (`NotFoundError`), and hardware locks (`NotReadableError`), confirming informative toasts and status feedback.
- **Voice Interaction & Speech Grounding:**
  - Speech recognition uses the Web Speech API (`webkitSpeechRecognition`), requiring a Chromium browser (Chrome/Edge) with physical microphone access.
  - Scene queries sent via the WebSocket relay execute against `PipelineOrchestrator` and return answers grounded in real detected scene objects.
  - Spoken output falls back to browser `window.speechSynthesis` when backend audio is absent.

## Step 25 Voice Assistant Reliability

Step 25 validated and enhanced voice query understanding and spoken answer reliability:
- Expanded query parsing in `query_reasoning.py` for natural speech-recognition phrasing, contractions (`what's`, `where's`, `i'm`, `can't`), user-perspective scene querying ("What is in front of me?"), nearby existence checks ("Is there a chair nearby?"), and lost item queries ("I cannot find my bottle").
- Added scene staleness detection: answers annotate when scene data is older than 30 seconds to prevent misleading users with outdated scenes.
- Maintained deterministic orchestrator query outputs without metric distance claims or safety hallucinations.

## Step 26 Spatial Reasoning Validation

Step 26 verified spatial consistency and accuracy across consecutive frames:
- **Viewer-Perspective Positioning:** Validated horizontal (`left`, `center`, `right`) and vertical (`above`, `middle`, `below`) classifications from the camera's perspective. Confirmed position updates dynamically as objects move across consecutive frames.
- **SceneMemory Position Preservation:** Fixed defect where `SceneMemory` omitted horizontal/vertical positions from object records, restoring full location descriptions in single-object queries.
- **Relative Depth:** Verified consistency of qualitative depth labels (`nearer_than`, `farther_than`, `similar_depth`) derived from Depth Anything V2. Confirmed that no metric distance ("meters") or safety claims ("safe to approach") are ever emitted.
- **Reciprocal & Symmetric Relations:** Updated `spatial_relationships.py` to emit reciprocal depth and overlap relationships, ensuring bidirectional consistency across the scene graph.
- **Tracking Continuity:** Verified on real video (`pinterest-video.mp4`) that ByteTrack continues tracked objects across consecutive frames with >=8 adjacent continuations. Fixed duplicate detection handling to skip self-relationships.
- **Dynamic Scene Consistency:** Verified that when objects move or disappear, old relationships are pruned and scene queries reflect the latest committed frame without using stale scene memory as current fact.


