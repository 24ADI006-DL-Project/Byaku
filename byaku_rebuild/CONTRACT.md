# BYAKU Rebuild API Contract

This document describes the implemented synchronous HTTP and browser WebSocket API. Browser frames use the existing frame-processing path. The browser owns its `getUserMedia` stream; WebSocket camera messages do not start the Python `LiveCameraProducer`. No background worker or frame queue is used.

## Health and Root

`GET /health` returns HTTP 200 while the API is serving requests:

```json
{"service":"byaku-rebuild-api","status":"running"}
```

`GET /` describes the live frame, query, and camera-capture API.

## Frame Processing

`POST /api/frame` accepts one JPEG image as base64. The declared dimensions must match the decoded image; `frame_id` must be unique and `frame_number` must increase beyond the latest committed frame.

```json
{
  "frame_id":"client-frame-1",
  "frame_number":1,
  "width":640,
  "height":480,
  "data":"<base64_jpeg>",
  "timestamp":1.0
}
```

A successful response has `type: "frame_result"`, the frame identity/size, pipeline status (`success` or `partial`), warnings, detections, spatial objects/relationships, graph, memory, scene description, and speech summary where produced. Detection, depth, and orchestration reuse the existing adapters and `PipelineOrchestrator`.

## Query

`POST /api/query` takes `{"query":"Is the bottle currently visible?"}`. The `query_result` response reflects the latest committed scene, or `not_ready` before a valid frame exists.

## Camera Lifecycle

`POST /api/camera/start` accepts an optional camera source; the default is `0`:

```json
{"source":0}
```

The response is `{"type":"camera_status","status":"started","source":0,"next_frame_number":1}`. Repeated start for the active source returns `already_running`; requesting a different active source returns HTTP 409.

`POST /api/camera/capture` captures and processes exactly one frame synchronously. Success returns the actual frame result:

```json
{
  "type":"camera_capture_result",
  "status":"success",
  "frame_id":"<producer-id>:1",
  "frame_number":1,
  "frame_result":{"type":"frame_result","status":"success"}
}
```

`partial` pipeline results are returned as `partial`, not upgraded to success. Frame IDs include a producer-specific token. Frame numbers continue above the latest committed API frame, and each capture attempt receives a strictly increasing number.

`POST /api/camera/stop` stops the producer and releases its capture. Repeated stops are idempotent. API shutdown also stops and releases an active producer.

Camera lifecycle and capture requests share a process-local nonblocking guard. An overlapping operation returns HTTP 409:

```json
{"type":"error","stage":"camera","code":"camera_busy","message":"a camera operation is already in progress","recoverable":true}
```

## Errors and State

Request validation errors return HTTP 422 with `code: "invalid_request"`. Existing frame-route validation and processing errors use the structured API error shape. Camera errors use `stage: "camera"` and codes `camera_busy`, `camera_already_running`, `camera_unavailable`, `camera_open_failed`, `camera_not_running`, `camera_read_failed`, `camera_encode_failed`, `frame_processing_failed`, or `camera_stop_failed`, depending on the failure.

Camera-open failures do not start capture. Read failures are reported; the producer stops and releases after its configured consecutive-failure limit. Processing/submission failures stop and release the producer. They do not replace the last committed scene; a later start can retry with a higher frame number. Capture processing is synchronous and can block the request until inference completes.

The overlap guard is process-local and does not coordinate multiple server workers. Browser camera messages use the existing Node relay and WebSocket protocol; the browser continues to own its `getUserMedia` stream. Node/browser source files and the UI theme are unchanged. No background processing or additional queue is implemented.

## Browser WebSocket

The Node relay forwards same-origin `/ws` traffic to the FastAPI `/ws` endpoint without translating messages. A connected client receives a `status` message. The supported browser messages are:

```json
{"type":"start_camera"}
{"type":"stop_camera"}
{"type":"ping","ts":42}
{"type":"query","text":"Is the bottle currently visible?"}
```

`start_camera` and `stop_camera` acknowledge the browser capture session only. They never open or stop the Python camera producer. `ping` returns `{"type":"pong","ts":42}`.

A browser frame has the existing canvas/JPEG shape:

```json
{
  "type":"frame",
  "frame_id":1,
  "width":640,
  "height":480,
  "data":"<base64_jpeg>"
}
```

The server requires a positive, increasing numeric client `frame_id`, positive integer dimensions, and base64 JPEG data whose decoded dimensions match. It assigns a unique internal session-scoped frame ID and a monotonically increasing server `frame_number`, then passes the resulting `FrameRequest` through the existing frame-processing path. The `frame_result` response echoes the numeric client `frame_id` for browser correlation and adds the server frame number. Its objects, detections, relationships, graph, memory, and scene description are actual pipeline output. When a valid relative-depth map exists, `depth_preview` contains a compact base64 JPEG colorized from that map for display; it is not the numerical map used for reasoning. No preview is returned when depth is unavailable. No events, timing metrics, or audio are synthesized.

The isolated `browser_protocol_adapter.py` compatibility layer adds frontend aliases from returned evidence only: `tracks` contains detected records that have a real `track_id`; relationship records retain their `subject`/`object`/`relation` fields and also expose `source`/`target`/`relationship`; and `scene_narration` is copied from the actual `scene_summary_speech.spoken_text` when available. The actual `depth_preview`, when present, passes through unchanged. If source data is absent, aliases are omitted or empty; confidence, distance categories, events, and timings are not invented.

`query` maps `text` to the existing query processor. The `query_result` includes the actual answer, matched objects, evidence, speech response, and a `narration` alias of `speech_response.spoken_text` when available. No audio field is generated.

Malformed JSON/messages, invalid frames, unsupported types, busy processing, and inference failures return a structured `error` message. Frame-related errors use `stage: "camera"` so the current browser client can clear its pending-frame state. A disconnect removes only that WebSocket session; it does not alter the committed scene or stop an unrelated Python camera producer.

## Dashboard Status

`GET /api/status` returns `backend`, current `camera` and `stage_status`, `last_frame`, `last_query`, `last_reasoning`, and `last_narration`. `last_frame.depth_preview` is present only when the last committed frame has a valid relative-depth map. Missing processing values are marked `unavailable` or `null`. Timing, FPS, events, and TTS audio are omitted because the rebuild does not produce them.

The same compatibility layer maps actual query `matched_objects` to `reasoning.objects` and `speech_response.spoken_text` to `narration`. It does not create dashboard-only `target`, metric distance, FPS, or timing values.

WebSocket sessions, frame numbering, and the fail-fast processing guard are process-local. Multiple server workers do not share that state. Inference runs synchronously in the server's thread pool; the browser currently waits up to five seconds for a frame result and does not queue frames.