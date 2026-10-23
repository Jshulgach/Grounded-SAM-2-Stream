# Performance update

Measured locally on 2026-09-17, RTX 3070 Ti (8 GiB), CUDA Torch 2.12.0+cu126. These short fixtures are regression checks, not a general detection-accuracy benchmark.

## Enabled on this PC

- Grounding DINO remains the default detector, with a 640-pixel shortest input edge instead of its native 800. Source previews retain their original configured size; output boxes are still normalized to the exact preview frame.
- The C920 and recorded-video source sample at up to 3 FPS, previously 1 FPS. The unavailable gripper source was not changed.
- Visible browser polling waits 150 ms after each completed update, previously 700 ms. Hidden tabs and connection failures wait 1500 ms. Requests do not overlap.
- Preview polling requests only the newest observation per source. Export still fetches the full retained history. No tracking or interpolated boxes are introduced.
- Existing one-frame source mailboxes, bounded result retention, cancellation and inference watchdogs remain in use.

Package defaults remain compatible. Operators opt into a smaller input through detector.image_size; omitting it keeps the native Grounding DINO size. Source sample_fps controls acquisition independently.

To reproduce the local balanced settings:

```json
{
  "sources": [{"id":"desk","name":"Desk","kind":"webcam","uri":0,"sample_fps":3,"max_width":960}],
  "detector": {"device":"cuda","image_size":640}
}
```

Remove image_size and set sample_fps back to 1 to restore the original profile. Higher resolution may still help with small or distant objects.

## Same-frame detector comparison

tools/benchmark_detector.py reads 12 frames at 250-ms intervals from the bundled deer video. Each case gets a first-frame measurement and two warmups before the reported median. Blank and wrong-object cases are included, along with multiple phrases and switching back to an earlier phrase. No camera access or image recording occurs.

| Case | DINO native median | DINO 640 median | YOLOE-26 small median |
|---|---:|---:|---:|
| Deer | 191.25 ms | 167.58 ms | 23.62 ms |
| Fire truck (absent) | 187.61 ms | 149.15 ms | 23.00 ms |
| Blank image | 199.94 ms | 149.70 ms | 21.64 ms |
| Deer + fire truck | 205.17 ms | 150.29 ms | 23.07 ms |
| Return to deer | 199.73 ms | 148.84 ms | 22.66 ms |

Both DINO configurations returned detections on 12/12 positive frames and 0/12 absent/blank frames. YOLOE returned detections on 6/12 positive frames and 0/12 absent/blank frames. DINO threshold was 0.35; YOLOE used 0.25 (at 0.35 it also missed the initial positive smoke frame). Scores are not calibrated across models. This fixture is insufficient to choose thresholds or establish general recall.

Therefore YOLOE is integrated as an **optional experiment**, not the local default. Its warm speed is promising, but the tested configuration did not meet the positive-fixture gate. No automatic fallback hides model failures or silently changes the detector.

## Multi-source path

At 3 FPS, six-second balanced searches processed 17 frames in the positive case, 16 in the negative case, and 31 total across two recorded sources (16/15). The two-source median inference was 169.09 ms; maximum recorded frame age was 531 ms. Capture startup, scheduling and model throughput mean the requested rate is a ceiling.

Timing-only JSON reports are in the ignored .work directory. The benchmark tools can recreate results without private cameras:

```sh
python tools/benchmark_detector.py --config YOUR_RECORDED_CONFIG.json --output detector-results.json
python tools/benchmark_video.py --config YOUR_RECORDED_CONFIG.json --duration 6 --output runtime-results.json
```

## Optional YOLOE setup

Use a separate virtual environment. Install a matching CUDA Torch/torchvision pair first, then requirements-yoloe.txt and the package/service. Check torch.cuda.is_available(): a generic dependency install can replace a CUDA pair with CPU wheels.

```sh
python -m pip install -r requirements-yoloe.txt
python -m pip install -e ".[service]"
python tools/prepare_yoloe.py
grounded-stream validate --config examples/yoloe.json
grounded-stream serve --config examples/yoloe.json --port 8766
```

The explicit setup tool downloads two official assets and verifies SHA-256. The adapter requires local checkpoint and text-encoder paths, disables automatic dependency installation/download checks, caches the current prompt embeddings, and reports weight hashes. It never saves model previews or crops. Missing assets fail before capture starts. Only trusted operator configuration can select model paths; HTTP clients cannot.

Dependencies are pinned to Ultralytics 8.4.155 and the CLIP revision in requirements-yoloe.txt. The [official YOLOE documentation](https://docs.ultralytics.com/models/yoloe/) explains custom text prompts and checkpoint variants. Its optional dependencies/weights carry their own terms; retain their notices when distributing that backend.

The prepared experiment on this PC uses .venv-yolo and .work/models. Grounded Stream on port 8765 continues to use .venv and Grounding DINO. No Homebase integration or firmware changes are part of this update.


## Live UI and packaging verification

- C920 continuous test: 2.83 sampled frames/sec, 142.06 ms rolling median inference, 172 ms median frame age at the measured checkpoint. This is processed-frame rate, not a browser display-FPS measurement.
- Latest-only snapshot: 1,309 bytes versus 104,205 bytes for retained history at the same checkpoint (98.7% smaller per response). Both use the same exact-frame evidence URLs.
- Browser Stop cancelled after 291 frames. No console errors. Export downloaded successfully and contained all 128 retained observations, rather than the single latest observation.
- All 103 tests passed in both development and the isolated installed-wheel environment. Lint, formatting and distribution builds passed.
- The optional YOLOE adapter also completed positive/negative/two-source jobs through real spawned workers. Cold prompt preparation was approximately 2-3.2 seconds and can dominate a short search; warm two-source median inference was 32.19 ms in that run. Warm detector timings alone do not describe startup responsiveness.

No household camera images were saved during these checks. A metadata-only test export was saved to Downloads by the browser. Multi-hour stability, broad household accuracy and live gripper-camera performance are not established by these tests.
