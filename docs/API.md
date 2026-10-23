# API v1

All API responses carry schema_version "1" where applicable. JSON field additions are compatible; changed semantics require a new schema version. The service listens on 127.0.0.1, one process only. Do not run multiple ASGI workers against the same cameras.

## Create a search

POST /v1/searches, Content-Type: application/json:

```json
{"targets":["red backpack","toy"],"source_ids":["desk"],"duration_seconds":10,"timeout_seconds":180}
```

Returns 202 with a search snapshot. Targets are literal phrases (1-8, <=120 characters each, <=400 total after deduplication). Sources must be unique configured IDs. Omitting duration_seconds retains the 10-second default for API compatibility. A numeric capture duration is 1-120 seconds; its total deadline is 5-600 seconds and includes queueing and model startup. The total deadline must be >= capture duration. Capture stops at the end of its window; an inference already in progress may finish afterward, within the overall deadline. A source marked stopped in a terminal snapshot was closed at search completion. Requests are limited to 8 KiB. Unknown fields, booleans as numbers, NaN and infinity are rejected.

For continuous search, send `"duration_seconds": null`. It runs until cancellation, all sources end, or a failure occurs. `timeout_seconds` still bounds queueing/model startup and each inference operation, but does not limit a healthy running session. The UI defaults to this mode; the CLI uses `--continuous` (its default remains a 10-second timed search). Cameras are never started just by loading the page. Closing a browser tab does not cancel server work; reconnect and use Stop, DELETE the search, or stop the server.

Status: queued -> loading -> running -> completed / partial / failed. cancelled and timed_out are terminal from any active stage. A completed search can contain no detections. Partial means at least one source succeeded while another was unavailable or stale. These are observations, not guarantees that an object is present or absent.

One detector serves all selected sources in round-robin order. Overlapping searches queue serially (four pending maximum). A continuous session occupies the detector until stopped; pending requests still expire at their own deadlines. Full queue returns 429 with Retry-After. DELETE is idempotent for a retained job; cancellation is acknowledged immediately and native processes are then terminated if necessary.

## Snapshot

Contains id, request, created_at, finished_at, per-source states and frame counts, processed_frames, observations, error and metrics. Observations include:

- source_id and frame_id: source identity and frame identity **within this search**.
- captured_at and processed_at: UTC wall times; frame_age_ms: monotonic processing age.
- media_time_ms for recorded video: playback position. captured_at is decode time, not original recording time.
- original_width / original_height and processed width / height.
- detections: label, score, box [x1,y1,x2,y2] normalized to the processed image.
- model, model_revision, device, inference_ms and peak_vram_mb (allocated GPU memory, not total device usage).
- evidence_url with required frame_id parameter.

Scores are model outputs, not calibrated probabilities. Results retain the latest 128 observations and 128 latency samples per job. Total processed_frames includes observations already removed from that window. Running snapshots report elapsed time and rolling median inference time. Live sources without fresh frames for five seconds are marked stalled; new frames restore running status.

Preview clients may request GET /v1/searches/{id}?latest=true to receive only the most recent retained observation per source. Counts and status remain unchanged. Omit the parameter for full retained history, including exports. capabilities.latest_snapshots advertises this option; capabilities.detector_backend identifies the configured adapter.

## Evidence consistency

Only the most recently processed JPEG per source is retained. Fetch the exact evidence_url from an observation. If another frame replaced it, the endpoint returns 409 rather than returning a different image. Fetch a fresh snapshot and retry. Missing or evicted evidence returns 404. Never overlay a box on an unrelated camera stream frame.

Preview bytes are capped at 32 MiB across all jobs. History is capped at 16 searches, retained for at most 15 minutes after completion; old terminal jobs may be evicted earlier to admit new work. All state is lost on service restart. JSON exports include relative preview URLs that will expire, not archival images.

## Events and errors

SSE emits event: search with a full snapshot when it changes, a heartbeat comment otherwise, and closes after a terminal snapshot. A reconnect receives the latest snapshot; there is no historical replay. An expired search emits event: expired. Clients can poll GET instead.

Common status codes: 400 invalid search, 401 token missing/wrong, 403 cross-origin, 404 absent/expired, 409 newer preview, 413 body too large, 415 wrong media type, 429 busy. Infrastructure failures appear in terminal search errors.

Configuration paths and camera addresses are omitted from public source metadata. Optional bearer token authentication applies to all /v1 endpoints. No CORS permission is granted; a future dashboard must use its own server-side authenticated adapter. The API does not open arbitrary URLs supplied by search clients.
