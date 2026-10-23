# Grounded Stream

**Local object search across cameras and recorded video.** Type `red backpack`, choose sources, and inspect detections in a standalone browser workspace.

This is the independent Grounded-SAM-2-Stream project. Version 0.2 replaces the experimental monolithic demo with an installable package, bounded worker processes, a CLI, and a local HTTP API. It currently performs **Grounding DINO object detection**. The historical repository name does not imply that SAM segmentation, tracking, speech recognition, or identity recognition is implemented.

## What works

- Direct object phrases, without an LLM or cloud inference account.
- Recorded video, USB webcams, and HTTP(S) MJPEG sources; up to eight configured sources.
- Continuous or timed searches, cancellation, inference watchdogs, per-source status, and partial results.
- Local browser UI with exact-frame boxes, keyboard-accessible controls, JSON export, and narrow-screen layout.
- CUDA or CPU inference, lazy model loading, normalized boxes, model revision and timing metadata.
- Versioned HTTP endpoints and server-sent events for future clients.
- Optional local Ollama phrase suggestions as a Python adapter; suggestions require review before searching.

No microphone opens automatically. Camera sources open only after an explicit search. There is no automatic frame recording or historical index. Model files are cached on disk; search metadata and previews are held in bounded memory. An explicit JSON export saves metadata, not image bytes.

## Quick start

Requires Python 3.11 or newer. Python 3.11 on Windows is the locally validated development environment; CI also defines Linux checks. Start with the included video before configuring a camera.

```sh
git clone https://github.com/Jshulgach/Grounded-SAM-2-Stream.git
cd Grounded-SAM-2-Stream
python -m venv .venv
```

Activate: Windows PowerShell `.venv\Scripts\Activate.ps1`; Linux/macOS `source .venv/bin/activate`.

Install a matching PyTorch/torchvision pair for your machine using the [official PyTorch installer](https://pytorch.org/get-started/locally/), then:

```sh
python -m pip install -e ".[vision,service]"
grounded-stream validate --config examples/video.json
grounded-stream serve --config examples/video.json
```

Open **http://127.0.0.1:8765**. Enter `deer` and start a search. The UI defaults to continuous mode; enable **Timed search** for a duration dropdown. Continuous camera searches keep running until Stop, including when the tab closes. Refresh reconnects to the active search. Recorded sources finish at end of file. First use downloads public Grounding DINO weights from Hugging Face; no account is normally required. Allow extra time for a cold download. The sample config pins the model revision.

The UI and validation commands do not load the detector. `pip install .` installs the lightweight base package; vision and service dependencies are optional.

### CLI

```sh
grounded-stream doctor
grounded-stream search --config examples/webcam.json --target mug --continuous
grounded-stream search --config examples/video.json --target deer --duration 10 --timeout 600
grounded-stream search --config examples/video.json --target deer --target "fire truck" --output result.json
```

Existing output files are never overwritten. Exit code 0 means every selected source supplied usable frames; 2 includes partial, failed, or timed-out searches. Finding zero objects is a valid completed search, not a failure. Ctrl+C cancels and cleans up workers.

### Your cameras

Copy `examples/webcam.json` to `config.local.json` and set the USB index, or adapt `examples/network.json` with your MJPEG URL. The network example uses a documentation-only placeholder.

```json
{
  "sources": [
    {"id":"desk","name":"Desk camera","kind":"webcam","uri":0,"sample_fps":1,"max_width":960},
    {"id":"gripper","name":"Gripper view","kind":"mjpeg","uri":"http://192.0.2.1:81/stream","sample_fps":1}
  ],
  "detector": {"device":"auto","threshold":0.35,"text_threshold":0.25}
}
```

Run `grounded-stream serve --config config.local.json`. File paths resolve relative to the configuration. Source IDs are stable and independent of filenames. API requests reference these IDs and cannot supply arbitrary camera URLs or filesystem paths.

Keep one owner per USB camera: close another browser/desktop preview before starting a search on that device. MJPEG reconnects with bounded backoff. RTSP, authenticated camera URLs, and browser-captured frame uploads are not supported yet.

### Offline use

After the model has downloaded, set `"local_files_only": true` and retain the same model/revision. Missing cached weights produce a failed search; this setting does not silently download them. There is no model preload during service startup.

## API

Loopback only. See [the API contract](docs/API.md) and local `/openapi.json`.

| Method | Endpoint | Purpose |
|---|---|---|
| GET | /v1/health | Worker and source configuration status |
| GET | /v1/capabilities | Explicit supported features and limits |
| POST | /v1/searches | Create a continuous or timed search |
| GET / DELETE | /v1/searches/{id} | Inspect / cancel |
| GET | /v1/searches/{id}/events | SSE snapshots until terminal |
| GET | /v1/searches/{id}/evidence/{source}?frame_id=â€¦ | Exact latest processed JPEG |

Set `GROUNDED_STREAM_TOKEN` before starting the service to require a bearer token for every `/v1` endpoint. Enter it in the UI's Connection panel; it is not saved to local storage. Host and origin checks restrict browser access. Do not publish this development service directly to a LAN or the internet.

## Development and validation

```sh
python -m pip install -e ".[service,dev]"
python -m pytest -q
ruff check src tests tools llm
ruff format --check src tests tools llm
python -m build
```

requirements-dev.lock pins the resolved development/service dependency set. To reproduce it, install with `python -m pip install -r requirements-dev.lock`, then `python -m pip install -e . --no-deps`. Model/GPU installation remains separate because platform-specific Torch builds differ.

For local troubleshooting, set `GROUNDED_STREAM_DEBUG=1` to print model exception tracebacks in the server terminal. Debug output can include local paths or query details; review it before sharing.

Tests use spawned fake workers rather than opening personal devices or downloading weights. With the vision extra, a generated-video codec test also runs. An explicit real-inference benchmark is separate:

```sh
python tools/benchmark_video.py --config examples/video.json --duration 5 --output benchmark.json
```

It checks a positive phrase, a negative phrase, and two video sources using the same detector process. It rejects live-camera configurations. See [validation evidence](docs/VALIDATION.md) for measured results and remaining gaps.

## Project direction

See [measured performance improvements and optional YOLOE setup](docs/PERFORMANCE.md).

See [model options and related projects](docs/MODEL_OPTIONS.md) for the next performance experiment.

Read [architecture and adoption gates](docs/ARCHITECTURE.md) and [migration from the old demo](docs/MIGRATION.md).

The standalone service comes first. Homebase should later consume its API through an adapter while this repo owns capture, inference, and model dependencies. Speech, historical search, SAM segmentation/tracking, and robotics must be separately tested additions.

## Credits and licenses

Built on [Grounding DINO](https://github.com/IDEA-Research/GroundingDINO), [Transformers](https://huggingface.co/docs/transformers/model_doc/grounding-dino), and ideas from [Grounded-SAM-2](https://github.com/IDEA-Research/Grounded-SAM-2). The optional phrase adapter uses [Ollama](https://docs.ollama.com/capabilities/structured-outputs).

Existing LICENSE, LICENSE_groundingdino, and LICENSE_cctorch notices are preserved. Model weights and optional future backends have their own licenses; review them for your intended distribution. Historical GIFs under assets illustrate the old experiment, not the current UI or proof of tracking.
