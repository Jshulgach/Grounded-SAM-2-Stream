# Standalone validation

Initial local validation: 2026-09-17. This is a development candidate; see the later checkpoint below for fresh verification.

## Executed checks

- Dependency-light base import and CLI: no Torch, OpenCV, Transformers, FastAPI or Ollama imported by CLI startup.
- Built both sdist and wheel; installed into a separate environment without access to global site-packages.
- Clean environment successfully runs doctor and configuration validation without vision dependencies.
- Spawned-worker regression suite covers bad configuration, malformed requests, exact source IDs, empty detections, queue saturation, cancellation during blocked model work, subsequent recovery, source failure/EOF, stalled capture, source fairness, bounded previews and result retention.
- API checks cover source-address omission, body size/content type, origin/host boundary, optional token authentication, exact-frame evidence conflict, missing/expired jobs and SSE terminal snapshots.
- Optional Ollama adapter checks use fakes: exact model tags, structured output validation, original query preservation and event-loop responsiveness.
- Real generated-video codec check verifies EOF cleanup and aspect-preserving resize.
- Browser check: real positive detections and matching boxes, no browser console errors; desktop and 390px-wide layout inspected. Missing-source validation, real no-match completion and manual cancellation were exercised in the browser.

The initial 77-test suite passed in both the editable GPU development environment and the clean installed-wheel environment. After the continuous-search update, all 88 tests passed in development, and the 19 continuous-search/API tests passed again against the rebuilt wheel in the isolated environment. Ruff and distribution builds passed. Two deprecation warnings originate in Starlette's test client (httpx and AnyIO compatibility); they do not indicate a failed application test.

## Real GPU benchmark

Hardware: NVIDIA RTX 3070 Ti, 8 GiB VRAM. Python 3.11.9, Torch 2.12.0+cu126, torchvision 0.27.0+cu126, Transformers 5.17.0, OpenCV 5.0.0. Inference uses scoped CUDA fp16.

Model: IDEA-Research/grounding-dino-tiny, revision a2bb814dd30d776dcf7e30523b00659f4f141c71.
Input: included immature_stag.mp4, 1920x1080 decoded and resized to 960x540, 1 sample/sec/source. Each capture window is 5 seconds.

| Case | Frames | Detections | Median inference | p95 inference | Total search |
|---|---:|---:|---:|---:|---:|
| Deer, first cached-model search | 5 | 5 | 184.22 ms | 699.36 ms | 14.13 s |
| Fire truck, same detector | 5 | 0 | 182.49 ms | 254.40 ms | 5.03 s |
| Deer, two concurrent video sources | 10 | 10 | 189.97 ms | 200.44 ms | 5.03 s |

Peak allocated GPU memory: 1923 MiB. Maximum processed-frame age in the two-source case: 453 ms. The supervisor's in-process health call stayed below 0.21 ms at p95 during these samples; this is not an HTTP latency benchmark.

The uncached first run, including downloading approximately 689 MB of weights, took 124.7 seconds. Startup and inference are reported separately because warm latency does not describe first-use experience.

These short examples verify execution and source fairness, not general detection accuracy or multi-hour stability. Confidence scores are not calibrated correctness probabilities. Two sources used the same recorded video, not two physical cameras.

## Clean CPU installation

A separate virtual environment without global site-packages installed the distribution and vision extra using Torch 2.14.0+cpu and torchvision 0.29.0. Real inference detected the deer on one frame: 8909 ms inference and 18.5 seconds total including cached-model startup. This was a functionality smoke test, not a warmed CPU benchmark. A finite capture window stops new acquisition while an already-started inference may finish within the overall deadline.

The GPU development environment reused the PC's installed CUDA Torch through system-site-packages; the independent CPU environment verifies that hidden machine-wide packages are not required. The GPU environment emitted a warning from an older global SciPy about NumPy compatibility; the clean CPU environment did not.

## Bugs reproduced and corrected

- Empty processor output: Transformers 5.17 returned zero boxes/scores but text_labels [""]; the old strict length check treated this as failure. The adapter now accepts that specific empty result and rejects other inconsistent lengths. Positive/negative/multi-source benchmark repeated successfully.
- Cancelled queued jobs could remain referenced after retention eviction. Queue admission and the supervisor now skip removed jobs; active cleanup is protected from eviction.
- Cancelled work could be marked loading again; interruption is checked inside the state transition.
- Preview replacement could produce mismatched boxes; evidence URLs include the frame ID and return 409 when superseded.
- A stalled camera with one old frame could be called healthy; final source freshness now affects completion.
- Legacy source labels depended on path underscores; configured IDs now define source identity.
- Legacy model tag matching, mandatory LLM rewrites, import-time GPU state and inaccurate package metadata were replaced.

## Not yet claimed

Live USB/MJPEG unplug/reconnect on the user's devices, multi-hour soak, Linux runtime execution, macOS inference, actual Ollama server/model quality, microphone transcription, historical indexing, SAM segmentation/tracking, person identity, Homebase integration, container device pass-through and robot actions remain outside this validation.

CI configuration is included but has not run on GitHub. Existing media/license provenance should be checked before a public release. The next gate is a supervised one-webcam/one-network-camera pilot.


## Continuous-search update

- Default browser mode is continuous; Timed search reveals 5/10/30/60/120-second options. The switch works with keyboard input and stays locked during a search.
- Explicit null duration enables continuous API sessions. Missing duration retains the existing 10-second API/CLI default. CLI adds --continuous.
- Regression coverage includes operation timeout validation, bounded rolling observations/latencies, hung model startup/inference, queue expiry while another search continues, EOF, stalled live sources, cancellation and recovery.
- C920 live search remained running at 183.97 seconds (past its old 180-second overall timeout): 174 processed frames, 128 retained observations, 128 latency samples, 176.84 ms rolling median inference, 30,418 retained JPEG bytes. This verifies bounded result/evidence buffers, not full-process memory stability over hours.
- A subsequent 5-second timed webcam search completed automatically with 4 processed frames and 184 ms median inference; inputs became available again. The workspace was left idle in continuous mode.
- Reloading the page reattached to the same search ID, target, selected source and Stop button. Stop then cancelled the search after 178 frames.
- Browser controls checked at desktop and 390px width with no horizontal overflow or console errors. Camera images were not saved; the ignored .work/continuous-pilot.json contains timing/count metadata only.
- Ruff lint/format checks and source/wheel builds passed. No alternate model was installed; see MODEL_OPTIONS.md for the researched benchmark plan.


## Performance integration

The later performance pass passed 103 tests in both the development and clean installed-wheel environments. It added compact preview snapshots, full-history export preservation, native-versus-reduced-size detector comparison, and an optional offline YOLOE adapter. See PERFORMANCE.md for the measured speed/accuracy tradeoff, live camera results and enabled local settings. YOLOE remains experimental because it missed half the sampled positive frames at the tested threshold.


## Checkpoint validation - 2026-10-23

- 103 tests passed in the development environment and again against the rebuilt wheel in the isolated validation environment.
- Ruff lint and formatting checks passed; source distribution and wheel built successfully.
- Installed-wheel doctor and example configuration validation passed without opening cameras or downloading models.
- Two upstream Starlette/httpx/AnyIO deprecation warnings remain. These are not test failures.
- The historical GPU timings above were not rerun for this checkpoint. Live camera recovery, multi-hour stability and Homebase integration remain separate validation gates.
- Local camera configuration, cached model files, development captures and virtual environments remain excluded from version control. Existing sample media remain unchanged; provenance review is still required before a packaged public release.
