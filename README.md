# BYAKU

BYAKU is a camera-based scene assistant. The browser UI handles camera access, voice/text interaction, and the developer dashboard. The FastAPI backend runs YOLO11n with ByteTrack and coordinates optional relative-depth and spatial scene reasoning.

## Start the application

From the repository root in PowerShell:

```powershell
& ".venv-1\Scripts\python.exe" "Backend\run_byaku.py"
```

The launcher starts FastAPI at `http://127.0.0.1:8765` and serves the current browser frontend from `UI/` through the Node relay at `http://127.0.0.1:3000`. Open `/` for the landing page, `/assistant.html` for the live assistant, or `/dashboard.html` for workflow status and Human Review. Stop the launcher with Ctrl+C.

Camera capture remains off until you choose **Start assistant** on the assistant page. Scene answers can be read aloud by browser speech synthesis when available; the backend cannot confirm audio playback.

The launcher uses the existing `.venv-1` environment and local Node.js installation, and expects the `UI/` relay dependencies to be installed. It does not install dependencies or download models. The active detector checkpoint is `Models/yolo11n.pt` (or the local trained checkpoint configured in code). Depth Anything V2 is used only when its model and dependencies are available locally. Set `BYAKU_INITIAL_FRAME_TIMEOUT_SECONDS` to configure the bounded initial-frame receive timeout (1–300 seconds; default 180).

## Tests

```powershell
& ".venv-1\Scripts\python.exe" -m pytest "Workflow\tests" -q -p no:cacheprovider
```

## Project structure

- `UI/` — browser frontend, static pages, shared assets, and the Node relay server.
- `Backend/` — FastAPI application, API/WebSocket routes, camera producer, persistent human-review feedback store, launcher, and timing support.
- `Workflow/` — detection/depth adapters, spatial reasoning, scene pipeline, dataset utilities, training tools, and maintained tests.
- `Models/` — active YOLO11n checkpoint and preserved custom-training checkpoints/runs.
- `Data/` — datasets, annotations, manifests, sample media, dataset provenance documents, and local human-review history.
- `Docs/` — concise project architecture and runtime notes.
- `Config/` — local Ultralytics configuration.
- `.venv-1/` — existing protected Python environment.
- `requirements.txt` — existing Python dependency manifest.

Dataset image IDs, annotation filenames, manifests, and generated checkpoint names remain unchanged where their pairings or training metadata depend on them.

See [Docs/API_AND_WEBSOCKET_CONTRACT.md](Docs/API_AND_WEBSOCKET_CONTRACT.md) for HTTP, browser WebSocket, status, timing, and Human Review details. The dashboard collects review feedback locally; it does not retrain the detector.
