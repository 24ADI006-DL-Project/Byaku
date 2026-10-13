# BYAKU Project Guide

**Purpose:** detailed, code-grounded reference for the BYAKU repository  
**Last reviewed:** 12 October 2026  
**Repository root:** `Byaku/`

## 1. Scope and reading this guide

BYAKU is a local, camera-based scene assistant. Its browser application requests
camera access only after the user chooses to start, sends image frames to a
local FastAPI service, and displays detections and scene reasoning. Browser
speech services can recognize voice commands and read returned descriptions
aloud. A separate workflow dashboard explains the processing stages and exposes
live status and human review.

This guide was prepared from the project source, tests, configuration, manifests,
and existing reports. “All files” needs an important boundary: this repository
also contains virtual environments, installed packages, compiled frontend
bundles, model weights, media, and large source-data archives. Those files were
inventoried where useful, but binary files, generated/vendor code, caches, and
installed third-party package contents are not reproduced or described line by
line here. They are not project source, and many cannot be meaningfully reviewed
as prose. The source inventory below covers the application, workflow, test,
documentation, and training code relevant to understanding BYAKU.

When an implementation statement differs from a report or an old UI, the
current runtime source takes precedence. The repository currently retains the
static browser pages in `UI/public/` and a legacy alternative layout, but the
standard launcher serves the current UI through the `UI/server/http_websocket_relay.js`
Node relay.

## 2. What a user experiences

The current browser frontend supports these paths:

| URL path | Page |
| --- | --- |
| `/` | Welcome screen, then the live assistant |
| `/assistant.html` | Live assistant |
| `/dashboard.html` | Project workflow and Human Review |

The welcome screen automatically moves on after three seconds; its Continue
button skips the wait. The home path then opens the assistant directly. The
assistant has a prominent microphone control: the user taps it to grant/start
browser speech recognition and says **“start”** to request the camera. Saying
**“stop”** turns the camera off when it is live. If browser speech recognition
is unavailable or denied, the page reports that and provides a manual camera
start button.

The microphone is not an always-on listener. Browser permission, browser
support, and a user gesture are required. Once the camera is on, the browser
sends frames for analysis. The UI displays detections and scene text, and
browser speech synthesis reads the first scene description aloud and subsequent
updates on a five-second cadence. A text question can also be submitted from the
assistant page and its spoken response replaces any ongoing routine narration.
Actual audio playback is controlled by the browser; the backend cannot confirm
that a person heard it.

The workflow dashboard is explanatory as well as operational. It describes the
camera-to-narration stages, reports stage/service state and timing when
available, shows the last processed frame, allows a person to confirm or correct
a detection, and retains an audit history with undo.

## 3. Architecture and request flow

```text
Browser (static UI served by the Node relay on 127.0.0.1:3000)
  ├─ microphone → browser speech recognition → “start” / “stop”
  ├─ camera permission → local video → JPEG frames (WebSocket /ws)
  ├─ text question (HTTP POST /api/query)
  └─ browser speech synthesis ← spoken narration returned by the API
                       │
                       ▼
FastAPI on 127.0.0.1:8765
  validate/decode frame → relative depth → YOLO + ByteTrack
       → remembered exact-crop correction → scene pipeline
       → spatial relations → scene memory → graph → description
       → serialized frame result / query result
                       │
          ┌────────────┴─────────────┐
          ▼                          ▼
local JSON review history      Models / Data on disk
```

The Node relay proxies `/api`, `/health`, and `/ws` to FastAPI. The browser
therefore uses same-origin paths and does not need to know the backend port for
each call. Camera capture belongs to the browser in the normal assistant flow;
`LiveCameraProducer` and the `/api/camera/*` routes implement a separate
optional Python/OpenCV camera-capture path.

The WebSocket protocol uses one JSON text message per operation. A new session
first receives status. The browser then sends `start_camera`, and submits
positive integer frame IDs, image dimensions, and base64 JPEG data. Frame
processing is synchronous from the caller’s perspective and returns a
`frame_result` or a structured error. The frontend allows one in-flight frame
at a time rather than building a queue of stale images.

## 4. Repository map

### Root

| Path | Responsibility |
| --- | --- |
| `README.md` | Short install/start/test entry point and top-level project map. |
| `requirements.txt` | Python runtime, model, server, and test package requirements. |
| `.gitignore` | Ignores Python caches, pytest cache, Node dependencies, and test outputs. |
| `.vscode/settings.json` | Selects the documented `.venv-1` Python interpreter for VS Code. |
| `Backend/` | FastAPI service, launch process, protocol adaptation, camera producer, feedback store, timing. |
| `Workflow/` | Vision adapters, scene reasoning, dataset/training workflows, and Python tests. |
| `UI/` | Current browser frontend and Node relay server. |
| `UI/` | Older static HTML/JavaScript UI, relay, assets, and Node tests. |
| `Models/` | Active base checkpoint and separate training runs/checkpoints. |
| `Data/` | Samples, dataset source/derived artifacts, manifests, reports, and local feedback. |
| `Config/` | Project-local Ultralytics and Matplotlib configuration/cache files. |
| `Docs/` | Architecture, API contract, and this guide. |

The root and project also contain `.venv` and `.venv-1`, `node_modules`, and
build/cache artifacts. They are environment or generated files, not modules to
edit. The documented launcher uses `.venv-1`.

### Backend modules

| File | Role |
| --- | --- |
| `Backend/run_byaku.py` | Starts FastAPI and the current UI relay; performs a bounded sample-frame startup warm-up; keeps the processes alive until interrupted. |
| `Backend/fastapi_application.py` | FastAPI routes, input models/validation, inference orchestration, WebSocket protocol, status, feedback endpoints, frame cache, structured errors. |
| `Backend/browser_message_adapter.py` | Adapts backend frame/query results for browser-facing fields without fabricating unavailable results. |
| `Backend/live_camera_producer.py` | Optional OpenCV capture producer; JPEG-encodes and submits frames through the existing frame-processing API; releases camera on failures/stops. |
| `Backend/feedback_store.py` | Locked, JSON-backed feedback history; schema versioning, idempotent submission, exact correction lookup, and audit-preserving undo. |
| `Backend/frame_timing_profiler.py` | Thread-safe stage timing samples and summary statistics. |
| `Backend/__init__.py` | Python package marker. |

On launch, `run_byaku.py` starts the backend on `127.0.0.1:8765`, builds
`UI/`, and serves it using the Node relay on `127.0.0.1:3000`. It then
attempts to send `Data/samples/bedroom_scene_desk.jpg` through the actual model
pipeline and ask a sample query before reporting the app ready. This optional
warm-up can take time on first model load. It seeds a retained example frame;
it does **not** grant browser camera access or turn on a live browser camera.
`BYAKU_INITIAL_FRAME_TIMEOUT_SECONDS` configures the bounded initial sample
frame wait (default 180 seconds; clamped to 1–300; non-integer input is an
error).

### Vision and reasoning modules

| File | Responsibility and important behavior |
| --- | --- |
| `Workflow/vision/object_detector_tracker.py` | Persistent Ultralytics YOLO inference plus built-in ByteTrack. Default weights are `Models/yolo11n.pt`; default confidence threshold is 0.25, IoU threshold 0.45, tracker config `bytetrack.yaml`. Validates boxes, IDs, confidence, and frame ordering rather than inventing missing track IDs. |
| `Workflow/vision/relative_depth_estimator.py` | Loads cached `depth-anything/Depth-Anything-V2-Small-hf` with downloads disabled; predicts normalized per-frame relative depth. Values are ordered higher = nearer; output is not metric distance. Model absence/failure is explicit. |
| `Workflow/reasoning/spatial_relation_engine.py` | Validates and clips XYXY boxes; derives center and image-region location; computes horizontal/vertical, IoU overlap, and optional depth relations. Untracked objects receive frame-local IDs. |
| `Workflow/reasoning/scene_relationship_graph.py` | Converts objects and supported relations into a NetworkX `MultiDiGraph`; preserves node/edge evidence and warns about malformed or dangling input. |
| `Workflow/reasoning/scene_memory.py` | Tracks recent observations and bounded history, distinguishes visible, temporarily missing, stale, and evicted objects, and prevents per-frame temporary IDs from being falsely carried between frames. Defaults: missing after 5 frames; keep 20 history observations; stale retention 300 frames. |
| `Workflow/reasoning/scene_description.py` | Describes currently visible objects and evidence-backed relations; rejects contradictory/unsupported relationships and excludes missing/stale items from present-tense claims. |
| `Workflow/reasoning/object_query_engine.py` | Parses supported natural-language object questions and answers only from current spatial/graph/memory evidence. It distinguishes visible, missing, stale, ambiguous, not found, unsupported, and not-ready states. |
| `Workflow/reasoning/narration_response.py` | Produces concise, structured speech text for a query or scene summary; removes duplicate/internal warning sentences and does not itself synthesize audio. |
| `Workflow/scene_pipeline.py` | `PipelineOrchestrator`: transactional coordinator for frame validation, spatial reasoning, staged memory, graph, scene description, narration, and later queries. It performs no camera capture or model inference itself. Failed/invalid frames do not replace the most recent valid scene. |

For image coordinates, incoming and outgoing detection boxes are pixel XYXY.
External `{width, height}` image dimensions are normalized to canonical
`(height, width)` in the reasoning code. Horizontal/vertical “left”, “right”,
“above”, and “below” describe image position, not a guarantee about a
three-dimensional world coordinate. Depth-based “in front”/“behind” is relative
and depends on usable depth output.

### Dataset, training, and media tools

| File | Responsibility |
| --- | --- |
| `Workflow/tools/dataset_validation.py` | Audits source material, YAML and YOLO labels; checks file/image-label pairs, finite normalized coordinates, duplicate hashes, provenance, group leakage and deterministic 70/20/10 source-group splits; writes manifests/reports and an annotation queue. |
| `Workflow/tools/prepare_detection_dataset.py` | CLI wrapper around the BYAKU source inventory, dataset preparation, and annotation-queue workflow. |
| `Workflow/tools/validate_detection_dataset.py` | CLI wrapper for writing/printing a dataset quality report. |
| `Workflow/training/prepare_open_images_training.py` | Creates an isolated dataset from the reviewed Open Images V7 manifest; checks source labels, attribution and licenses, duplicates, and prepares train/val/test image-label pairs. |
| `Workflow/training/run_detector_training.py` | Validates all splits, trains an isolated checkpoint under `Models/training_runs`, records a config/log/summary and hashes, and provides a test-split comparison/failure-preview evaluation path. |
| `Workflow/tools/import_visual_genome_dataset.py` | Streaming archive/record analysis, exact class mapping, box conversion, duplicate grouping, group splits, image-rights checks, and staged import. Current project report blocks importing its candidate images. |
| `Workflow/tools/dataset_pipeline.py` | Earlier phased dataset download/filter/subset/YOLO preparation workflow. Treat as a separate/legacy path; use the quality-gated current preparation path for the reviewed full training set. |
| `Workflow/tools/build_full_dataset.py` | Full-dataset build/download helper; retains a separate historical route. |
| `Workflow/tools/process_video_pipeline.py` | Offline video frame extraction/detection pipeline. |
| `Workflow/tools/visualize_labeled_samples.py` | Produces seeded labeled-sample visualizations. |

Package markers are `Workflow/__init__.py`, `Workflow/vision/__init__.py`,
`Workflow/reasoning/__init__.py`, `Workflow/tools/__init__.py`, and
`Workflow/training/__init__.py`.

### Browser application

`UI/public/` contains the current static browser frontend: `welcome/index.html`,
`assistant.html`, and `dashboard.html`, plus CSS and JS assets. The Node relay in
`UI/server/http_websocket_relay.js` serves those pages and proxies `/api` and `/ws`
traffic to FastAPI.

The assistant:

1. Requests speech recognition only when the user taps the microphone.
2. Recognizes “start” and “stop”; it does not continuously record in the
   background. Unsupported recognition and permission errors are surfaced.
3. On “start”, asks the browser for a video-only camera stream (no audio track),
   shows the local preview, and opens `/ws`.
4. Sends a JPEG frame no wider than 960 pixels about every 900 ms, only when the
   socket is open, current video data exists, and the previous frame has
   returned.
5. Displays frame detections and the scene narration. It speaks the first
   returned scene summary immediately and schedules later narration updates
   five seconds apart.
6. Supports typed questions at `/api/query`; their spoken answer takes priority
   over routine scene narration.
7. Stops timers, speech output, the WebSocket, and local camera tracks on stop,
   disconnect, or page unmount.

The workflow page polls `/api/status` and `/api/feedback?limit=20` every four
seconds. It displays stage status, the retained frame, processing and
initialization timing, the latest reasoning, and recent review actions. A
correction requires a non-empty corrected label. The guide card explains each
pipeline stage and links to Human Review.

Styles/assets include `UI/public/welcome/favicon.svg`, icons, `assets/Byakugan.jpeg`,
`assets/byaku-fire.mp4`, and the static welcome-page bundle. `UI/public/assistant.html`
and `UI/public/dashboard.html` are the live browser pages; the relay server handles
`/api`, `/health`, and `/ws` forwarders. `UI/package.json` declares the Node relay
and its browser tests.

### Current browser frontend

`UI/public/` contains a separate older static UI:

- `welcome/` is a built welcome page and bundled/minified assets.
- `assistant.html` and its CSS use `camera_preview.js`,
  `browser_websocket_client.js`, `assistant_speech.js`, and
  `assistant_interface.js`.
- `dashboard.html` uses `workflow_dashboard.js` and its legacy dashboard CSS.
- Additional scripts include `frontend_config.js` and `intro_animation.js`.
- `UI/server/http_websocket_relay.js` is a Node HTTP/WebSocket relay for that
  static public directory.
- `UI/package.json` defines the legacy Node relay and tests. Its two test files
  cover browser frame processing and speech/narration scheduling.

The standard `Backend/run_byaku.py` launcher starts the FastAPI service and then
uses the Node relay in `UI/server/http_websocket_relay.js` to serve the browser
pages. Changes to the static files in `UI/public/` therefore affect the frontend
shown by the standard launcher.

## 5. Inference and data contracts

### Frame processing

The HTTP frame model requires non-empty `frame_id`, a positive `frame_number`,
positive integer `width` and `height`, and non-empty base64 JPEG `data`.
Whitespace-only frame IDs are rejected. The WebSocket browser protocol uses a
positive integer browser frame ID and assigns the backend’s next monotonically
increasing frame number.

The frame path is:

1. Validate request and lock shared frame processing; busy work returns 409.
2. Initialize adapters once and measure initialization separately from the
   current-frame processing time.
3. Decode base64 and OpenCV image, and check declared and decoded dimensions
   match.
4. Estimate/validate an optional depth map. Non-finite, wrong-sized, or
   otherwise invalid depth does not silently become metric distance.
5. Run YOLO with persistent ByteTrack; keep validated class, confidence,
   bounding box, and track ID where available.
6. Look for an unused exact visual-crop correction and apply it only to the
   matching inference.
7. Run `PipelineOrchestrator` to compute spatial objects/relations, update
   scene memory, build the graph, describe the scene, and produce narration.
8. Serialize a `frame_result` with detections, scene objects, relations, graph,
   memory, scene description, speech text, warnings, stage status, timings, and
   a short-lived review-frame token. A depth preview is attached if available.

The latest successful frame, preview, and pipeline status are retained in
backend memory for status, scene questions, and review. Camera disconnect does
not mean this last frame is deleted; status separately reports whether a camera
is connected and whether any frame has ever been retained.

### Query processing

`POST /api/query` accepts `{"query": "..."}` with non-empty trimmed text. It
queries the most recently committed valid scene; it does not run a new camera
capture. The answer includes status, matched objects, evidence, warnings, and a
`speech_response`. The query engine has pattern-based supported intents,
including visible-object lists, location, relative-position, visibility,
last-seen and lost-object questions. It reports uncertainty/ambiguity instead
of inventing evidence. A scene older than 30 seconds can be identified as
stale.

### HTTP endpoint inventory

| Method/path | Meaning |
| --- | --- |
| `GET /health` | Liveness: service identifier and `running`. |
| `GET /` | API identity/description. In the browser app this path is served by the frontend instead. |
| `GET /api/status` | Backend/camera state, stage state, last frame/preview, latest query/reasoning/narration and available timing. |
| `POST /api/frame` | Validate/process one encoded frame through the shared pipeline. |
| `POST /api/query` | Ask about the latest valid scene. |
| `GET /api/feedback?limit=N` | Read up to 500 newest review records; default 100. |
| `POST /api/feedback` | Create an idempotent review record (201 on a new submission). |
| `POST /api/feedback/{feedback_id}/undo` | Mark a review record undone without deleting its audit record. |
| `POST /api/camera/start` | Start the optional Python/OpenCV camera producer. |
| `POST /api/camera/capture` | Capture/process one frame from that producer. |
| `POST /api/camera/stop` | Stop/release that producer. |
| WebSocket `/ws` | Browser camera session, frame, ping/pong, and query protocol. |

Validation, invalid state, and busy operations use explicit responses/errors.
The APIs do not promise to process a concurrent frame or query while another
one holds the shared pipeline lock.

## 6. Human-in-the-loop behavior and limits

The dashboard’s Human Review panel presents the last processed frame and its
detections. A person can confirm a prediction or submit a corrected class name.
The request includes the original prediction, frame references, tracking ID,
review token/dimensions when available, correction type, review status, and
submission ID. Corrections are persisted in
`Data/feedback/review_history.json`; duplicate submissions with the same
submission ID return the original record.

Where the backend still has the frame (the in-memory review-frame token is valid
for up to 180 seconds), a class correction can store a SHA-256 fingerprint of
the normalized detected-object crop. An exact matching crop and original class
can receive that correction once on a later inference; application is recorded
in the history. This is not a learned parameter update. **Review feedback does
not retrain, fine-tune, or change model weights.** Undo preserves the audit
record and its prior status.

Other correction types are accepted by the API, but the current dashboard
currently exposes per-detection confirmation and class correction; it is not a
full annotation editor for boxes, positions, or dataset export. The dashboard’s
feedback history is local JSON data, not a multi-user service/database.

## 7. Model, training, and dataset status

### Live model versus trained artifacts

The production runtime default is the original checkpoint
`Models/yolo11n.pt` (5,613,764 bytes in this checkout), followed by the configured
ByteTrack tracker. The detector resolves `DEFAULT_WEIGHTS` to this base
checkpoint. Separate fine-tuned weights exist under training-run directories;
their existence does **not** mean the live runtime uses them. Switching live
inference requires an intentional, tested runtime configuration change.

An Open Images fine-tuning run is recorded at
`Models/training_runs/yolo11n_byaku_openimages7_full600_20261011/`. The training
summary reports completion in 5,025.57 seconds, preserves `best.pt` and
`last.pt`, and records SHA-256 hashes. It used the base checkpoint; CPU; image
size 640; batch 16; workers 2; seed 42; deterministic mode; 20 epochs configured
with patience 5. Train/validation/test were kept separate; test was not used
during training. The base checkpoint hash before and after training is
identical. The run’s `results.csv`, logs, plots, config, summaries, checkpoints,
and validation previews are training artifacts, not the runtime selection.

An older separate run, `yolo11n_byaku_custom`, has training outputs and
checkpoints but should not be treated as the current reviewed dataset
experiment. No evaluation metrics/report for the newer trained checkpoint was
found in the inspected run directory; do not infer generalization quality from
“completed” alone.

### Reviewed Open Images dataset

The prepared dataset is `Data/datasets/prepared_open_images_v7_full600/`:

| Split | Images | Bounding boxes |
| --- | ---: | ---: |
| Train | 420 | 1,685 |
| Validation | 120 | 382 |
| Test | 60 | 210 |
| **Total** | **600** | **2,167** |

The split is deterministic and source-grouped; the preparation report records
zero cross-split exact duplicates, zero cross-split near-duplicate pairs at its
documented perceptual-hash threshold, zero manifest/label mismatches, and
per-image CC BY 2.0 attribution evidence for all 600 images.

The YOLO training map contains **seven trainable** classes:

| ID | Class | Total reviewed boxes |
| ---: | --- | ---: |
| 0 | `person` | 519 |
| 1 | `bottle` | 245 |
| 2 | `book` | 570 |
| 3 | `chair` | 337 |
| 4 | `table` | 277 |
| 5 | `mobile_phone` | 112 |
| 6 | `laptop` | 107 |

`bag` has zero eligible Open Images V7 bounding boxes. It is documented as
unsupported/no training examples, rather than synthetically annotated or
silently merged with handbag, luggage, or another class. The prepared `data.yaml`
has seven names; it does not contain a trainable `bag` entry.

Dataset sources are not interchangeable merely because files are present:

- Open Images V7 is the reviewed training source; its full 600-image report
  documents split, duplicate, attribution, and label checks.
- The Visual Genome report blocks image download/conversion because image-level
  rights/attribution are unresolved and many annotation/metadata URLs differ.
  Zero images were imported by that blocked plan.
- The COCO128 report blocks use in training because the local archive does not
  establish image-level Flickr licensing/attribution. Its structural inspection
  does not grant image rights or create a BYAKU dataset.
- `Data/datasets/byaku_detection/` supports a separate local-source inventory
  and manual-annotation workflow; it is distinct from the prepared Open Images
  training tree.

## 8. Data and configuration map

`Data/samples/` contains `bedroom_scene_desk.jpg`,
`bedroom_scene_window.jpg`, and `indoor_objects_video.mp4`. The launcher uses
the desk image for startup warm-up.

`Data/datasets/` contains source/derived data for:

- `byaku_detection/`: class mapping, source/split/dataset manifests,
  duplicates, annotation queue, and dataset validation artifacts.
- `open_images_v7/`: source investigation, quality reports, manifests,
  attribution, cached metadata and staged source material.
- `prepared_open_images_v7_full600/`: reviewed images/labels, seven-class YAML,
  split and attribution metadata, and preparation/quality reports.
- `coco128/`: the inspected source archive and validation report; not a
  cleared training dataset.
- `visual_genome/`: source archives and mapping/analysis/planning reports; the
  proposed image import is blocked.

`Data/documentation/` has the dataset investigation and quality-gate reports.
`Data/feedback/` holds the local review JSON file. Existing reports explain
provenance and stop decisions in more detail than the runtime code.

`Models/` contains the active root `yolo11n.pt` and isolated run outputs.
`Config/ultralytics/Ultralytics/settings.json` configures dataset, model, and run
directories for the project. `Config/training/ultralytics_settings/` contains
training-specific Ultralytics/Matplotlib settings and generated font/cache data.
The detector sets `YOLO_CONFIG_DIR` to the project-local location so Ultralytics
does not silently place project artifacts elsewhere.

## 9. Dependencies and local environment

The current documented interpreter is Python 3.11 in `.venv-1`. Root
`requirements.txt` declares the computer-vision/data stack (OpenCV, NumPy,
PyTorch, Ultralytics, NetworkX), FastAPI/Uvicorn/Pydantic/WebSockets, and
additional optional/later-stage speech and UI packages. The active browser UI is
`UI/public/`, and the `UI/server/http_websocket_relay.js` Node relay declares `ws`.

The launcher expects those environments and dependencies to already exist. It
does not install Python/npm dependencies and does not download the YOLO
checkpoint or Depth Anything model. Depth Anything V2 is only loaded from local
cache with `local_files_only=True`; if that optional model cannot initialize,
depth failure is reported rather than downloading model data unexpectedly.

The app binds its local services to loopback (`127.0.0.1`), not all network
interfaces. `requirements.txt` expresses allowed version ranges; it is not a
fully pinned lockfile. `UI/package-lock.json` records the frontend dependency graph.

## 10. Run, build, and validate

From the repository root in Windows PowerShell:

```powershell
& ".venv-1\Scripts\python.exe" "Backend\run_byaku.py"
```

Open `http://127.0.0.1:3000/`. Use Ctrl+C in the launcher terminal to stop the
application.

Run the current frontend relay/server tests from the repository root:

```powershell
node --test UI\tests\*.test.cjs
```

Run Python tests from the repository root:

```powershell
& ".venv-1\Scripts\python.exe" -m pytest "Workflow\tests" -q -p no:cacheprovider
```

Targeted tests can be passed explicitly before the full suite. Several model or
real-camera tests depend on local model/cache/assets and may be slower or
environment-sensitive; consult each test’s marks/implementation before
launching a costly hardware/model exercise.

## 11. Test coverage index

`Workflow/tests/` contains 33 Python files: 32 `test_*.py` modules and one
standalone relay integration-check helper. Their names indicate independent
contracts and integration coverage:

- **Service/API/protocol:** `test_fastapi_application.py`,
  `test_api_contracts.py`, `test_browser_message_adapter.py`,
  `test_websocket_messages.py`, `test_frontend_integration.py`,
  `test_live_application_integration.py`, `test_accessibility_support.py`.
- **Camera, launch, runtime and timing:** `test_camera_api.py`,
  `test_camera_capture.py`, `test_application_launcher.py`,
  `test_runtime_configuration.py`, `test_frame_timing_profiler.py`.
- **Human review:** `test_feedback_api.py`.
- **Models and performance:** `test_object_detector_tracker.py`,
  `test_relative_depth_estimator.py`, `test_real_application_inference.py`,
  `test_real_inference_image.py`, `test_run_byaku_yolo11n.py`,
  `test_pipeline_performance.py`.
- **Scene behavior:** `test_scene_pipeline.py`,
  `test_scene_memory.py`, `test_scene_relationship_graph.py`,
  `test_scene_description.py`, `test_object_query_engine.py`,
  `test_spatial_relation_engine.py`, `test_spatial_reasoning.py`,
  `test_narration_response.py`, `test_voice_interaction.py`.
- **Data/video/training preparation:** `test_dataset_validation.py`,
  `test_open_images_training_preparation.py`,
  `test_visual_genome_dataset_import.py`,
  `test_video_processing_pipeline.py`.
- **Relay helper:** `node_relay_integration_check.py` is a standalone
  integration-check script rather than a pytest-style `test_*.py` module.

The Node tests are `UI/tests/frame_processor.test.cjs` and
`UI/tests/assistant_speech.test.cjs`. They cover one-frame-at-a-time capture,
waiting for a frame captured after a question, cold-start watchdog behavior,
retaining encoded frame data for review, cancelling late encodes, voice choice,
speech priority, periodic scene narration, stale narration suppression, and
timer cleanup.

These suites verify specific contracts; passing unit tests does not itself
prove that a physical camera, browser microphone, model checkpoint, operating
system speech voice, or hardware acceleration is available on every machine.

## 12. Privacy, reliability, and known limits

- Camera video starts only after an explicit manual start or recognized
  “start” command, and the normal browser capture requests no audio media
  track.
- The user must grant camera/microphone permissions. Use a supported browser;
  browser speech recognition is not universal.
- Browser speech recognition and synthesis are platform/browser services. This
  project does not establish whether a specific browser performs speech
  recognition locally or remotely; check that browser’s privacy behavior.
- Camera frames travel from the browser to the local backend for inference.
  Recent frame data for Human Review is cached in backend memory temporarily;
  it is not written into review history. The status endpoint can retain a
  preview in memory after the live camera disconnects.
- Dataset files/manifests and model assets are local. Optional dataset and
  training scripts can download material when explicitly run; the application
  launcher itself does not install dependencies or download model checkpoints.
- Speech text is generated from current vision/model evidence. It can be
  uncertain, stale, or incorrect; it is assistance, not a safety guarantee.
- Depth is relative, not a tape-measure distance. 2-D left/right positions are
  positions in an image.
- Voice recognition currently interprets start/stop. Questions are reliably
  available through the visible text form; arbitrary spoken-question handling
  is not promised by the current browser assistant.
- Human correction memory performs a restricted exact-crop correction; it is
  not model learning, detector retraining, or a general visual similarity
  system.
- The UI has a graceful manual camera-start fallback when Web Speech
  Recognition is missing or permission fails.
- The custom-trained checkpoint is not selected by the production detector
  default; do not assume the deployed/live class vocabulary equals the
  seven-class training YAML until weights are explicitly changed and tested.

## 13. Existing documentation

- [README.md](../README.md) — concise project introduction, run/test commands,
  and folder summary.
- [PROJECT_LAYOUT.md](./PROJECT_LAYOUT.md) — short service/runtime map.
- [API_AND_WEBSOCKET_CONTRACT.md](./API_AND_WEBSOCKET_CONTRACT.md) — HTTP,
  WebSocket, status, timing, and review payload behavior.
- [DATASET_QUALITY_GATE_REPORT.md](../Data/documentation/DATASET_QUALITY_GATE_REPORT.md)
  and [DATASET_INVESTIGATION_REPORT.md](../Data/documentation/DATASET_INVESTIGATION_REPORT.md)
  — source preparation and quality-gate details.
- Dataset-specific reports under `Data/datasets/` — Open Images, COCO128,
  Visual Genome, source investigation, and workflow audit findings.
