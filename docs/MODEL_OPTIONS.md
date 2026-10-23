# Model options and related projects

Research checked 2026-09-17. Grounding DINO remains the default. YOLOE has since been integrated and benchmarked as an optional backend; see [measured performance](PERFORMANCE.md) for results and the accuracy gate.

## Related work

- [Grounded-SAM-2](https://github.com/IDEA-Research/Grounded-SAM-2) combines prompted detection with video segmentation/tracking. It is the closest research precedent; this repo focuses on a usable standalone service and camera lifecycle.
- [Frigate semantic search](https://docs.frigate.video/configuration/semantic_search/) searches tracked camera events using local embeddings. It is a useful reference for a future event library; our current search processes live frames and does not maintain historical events.
- [NVIDIA NanoOWL](https://github.com/NVIDIA-AI-IOT/nanoowl) optimizes OWL-ViT with TensorRT, with examples and performance evidence on Jetson Orin. Its optimization approach is relevant, but its platform benchmarks do not predict Windows RTX 3070 Ti latency.

## What to benchmark next

| Candidate | Intended experiment | Main tradeoff |
|---|---|---|
| Current Grounding DINO tiny | Keep as baseline and fallback | Already validated here; approximately 170-190 ms inference in earlier short GPU runs |
| YOLOE-26 small, then nano | First alternate text-prompted detector | Measure small/rare objects, attribute phrases and false positives alongside speed |
| YOLO-World | Secondary open-vocabulary comparison | Additional runtime and packaging work |
| SAM 3 / 3.1 | Later experiment for masks and persistent video tracks | Larger model and separate environment; unnecessary for a boxes-only speed upgrade |

[YOLOE's official implementation](https://docs.ultralytics.com/models/yoloe/) supports text and visual prompts. Start with `yoloe-26s-seg.pt`, not the prompt-free `-pf` variant: the latter cannot accept custom classes. Reuse prompt embeddings for a search. Its first text-prompt call can install CLIP and download a text encoder; provision and pin those assets explicitly before allowing camera workers to run offline. YOLOE-26 needs Ultralytics 8.4.0 or newer. The [research code is AGPL-3.0](https://github.com/THU-MIG/yoloe); assess optional-backend distribution terms separately from this repo's existing license.

[YOLO-World](https://github.com/AILab-CVC/YOLO-World) offers a prompt-then-detect design with reusable vocabulary embeddings. It is a worthwhile second comparison if YOLOE's accuracy is insufficient.

[SAM 3](https://github.com/facebookresearch/sam3) adds concept-prompted detection, segmentation and tracking; SAM 3.1 adds multiplexed multi-object tracking. The documented setup requires Python 3.12+, Torch 2.7+, CUDA 12.6+, and access to gated weights. Use a separate environment from our Python 3.11 installation and measure 8-GB VRAM fit. It is not the first latency experiment.

## Controlled adoption plan

1. Add an optional detector adapter preserving current normalized boxes, labels, timing metadata, cancellation and worker isolation. Keep model dependencies out of the base install.
2. Pin packages and weights in a separate benchmark environment. Include cold startup, prompt preparation and offline startup.
3. Compare the same recorded fixtures and resolutions: mug, bottle, toy, tiny/distant object, low light, occlusion, empty scene and wrong phrase. Label expected boxes and count false positives/misses; scores across models are not comparable accuracy measures.
4. Measure warm p50/p95 inference, end-to-end frame age, VRAM, throughput and dropped frames with one and two sources. Test 1, 3 and 5 sample FPS; do not assume published hardware results transfer.
5. Repeat long-running searches, cancellation during inference, worker failure and camera outage recovery. Keep the previous backend selectable until these checks pass.
6. Only then expose a backend selector and consider Homebase adoption.

Continuous duration is independent of frame rate. The current local camera config samples at 1 FPS, so a faster model alone will not make its preview smoother. First measure higher sampling rates; later decouple preview refresh from inference while preserving exact-frame overlays.

A further architectural experiment is periodic detection plus lightweight tracking between detections. Tracks need their own age, identity and loss handling, with regular redetection; reusing a stale box on a new frame would be misleading.

The later performance update added an isolated YOLOE experiment; its results are in PERFORMANCE.md.
