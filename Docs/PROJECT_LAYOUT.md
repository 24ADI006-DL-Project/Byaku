# Project layout and runtime

Run the application from the repository root with `.venv-1\Scripts\python.exe Backend\run_byaku.py`. The launcher starts the FastAPI service and serves the static browser frontend from `UI/` through the Node relay on port `3000`; it does not install dependencies or download model files. `BYAKU_INITIAL_FRAME_TIMEOUT_SECONDS` controls the bounded initial-frame wait.

The service health endpoint is `/health`. The frontend serves the landing page at `/`, the assistant at `/assistant.html`, and the workflow dashboard at `/dashboard.html`; the relay proxies API and WebSocket traffic to FastAPI. Backend routes and message shapes are defined in `Backend/fastapi_application.py` and `Backend/browser_message_adapter.py`.

`UI/` contains the current browser app and relay server; `Backend/` owns the HTTP/WebSocket service and launch lifecycle; `Workflow/` contains scene processing, model adapters, reasoning, training utilities, and automated tests; `Models/` contains local model assets and training runs; `Data/` contains datasets and samples; `Config/` contains local runtime configuration. See [API_AND_WEBSOCKET_CONTRACT.md](./API_AND_WEBSOCKET_CONTRACT.md) for current routes and message shapes.
