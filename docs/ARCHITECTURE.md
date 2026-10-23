# Standalone architecture and adoption gates

## Current boundary

Browser UI / CLI -> SearchManager -> capture processes + one detector process.

- contracts.py: validated, dependency-light request/result values.
- config.py: operator-controlled source and model configuration.
- runtime.py: continuous/timed sessions, queue admission, operation deadlines, cancellation, fairness, retention.
- workers.py: killable native capture and inference boundaries.
- detector.py: lazy Grounding DINO adapter and factory, scoped precision, configurable inference size, normalized detections.
- yoloe.py: optional offline adapter, explicit provisioned assets, cached text prompts.
- service.py: loopback HTTP/SSE and packaged static UI.
- phrases.py: optional local structured phrase suggestion; not in the direct detection path.

There is no import-time model loading, implicit LLM request, automatic recording, or microphone capture. Each source has a one-frame mailbox. Slow inference drops obsolete frames instead of building an unbounded backlog. File playback is paced to its recorded frame rate. Blocking native camera reads cannot hold up HTTP request handling; the supervisor can terminate their process.

A cancelled/timed-out inference discards the detector process, so the next search reloads it. Successful searches reuse it. CUDA allocation can therefore remain while the service is idle; stop the service to release it. Maximum eight capture processes share one detector. CPU and GPU performance depend on source count and model choice; sample_fps is an acquisition ceiling, not guaranteed inference throughput.

This is a local development service, not a high-availability camera recorder. Restart loses results. There is no durable job store or user account system. Optional Ollama suggestions use a timeout-bounded HTTP thread; cancelling the awaiting task does not interrupt the underlying request before its timeout.

## Completed foundation to validate

1. Real Python package and wheel, optional dependencies, current model API.
2. Explicit configured sources and deterministic direct phrases.
3. Bounded runtime and local service with documented contracts.
4. Standalone UI, fake-worker regression suite, real recorded-video benchmark.
5. Reproducible model revision in the sample config and honest feature flags.

See VALIDATION.md for actual executed checks. A CI definition is not evidence of a completed remote run.

## Next standalone pilot

Use one webcam and one network camera, with explicit user initiation. Record time to first result, median/p95 processing age, wrong detections, missed detections, reconnect behavior, and camera ownership conflicts. Test an actual unplug/replug and network interruption. Use visible objects (toy, mug, backpack) rather than person identity. Add a small consented labeled fixture set before comparing model upgrades.

Only after that gate should the service become a Homebase dependency.

## Homebase adoption

Keep this repository independent. Homebase owns the dashboard, source display names/room mapping, assistant conversation, user preferences and integration credentials. Grounded Stream owns model environments, capture, searches and observation evidence.

A Homebase server-side adapter should:
1. Check /v1/capabilities and /v1/health.
2. Map an explicit search intent to configured source IDs and confirmed phrases.
3. Submit a finite search, poll or subscribe, and support cancellation.
4. Display source name, time and exact-frame evidence, with stale/partial states.
5. Store credentials outside client bundles.
6. Avoid competing camera ownership.

The current Homebase webcam preview is browser-local. Reusing those frames would require a separate explicit frame-sharing permission and API; this implementation does not upload them. A standalone service owning the USB camera is the simpler first pilot.

Do not infer a child's identity from a generic person detector. Object observations also do not establish 3D reachability for robot actions. Tracking identities, camera calibration, depth, robot planning and motion authorization are separate interfaces.

## Later modules, each with its own acceptance gate

- Push-to-talk transcription: local optional engine, visible recording state, editable transcript, no always-on microphone by default.
- Query suggestions: existing Ollama adapter plus explicit confirmation UI; direct object text remains usable if Ollama fails.
- Historical observations: opt-in retention, timestamps, deletion, disk quotas and consented recordings; no implicit JPEG directory crawl.
- Segmentation/tracking: backend interface and separate extras; evaluate current SAM options, licenses, VRAM and tracking drift before adding a default.
- Deployment: reproducible CPU/GPU container variants, model cache volume, explicit camera pass-through and health checks. Linux USB and Windows/WSL camera access need different validation.
- Community release: fresh-machine install, Linux CI result, license/provenance review for bundled media, tagged release only after approval.
