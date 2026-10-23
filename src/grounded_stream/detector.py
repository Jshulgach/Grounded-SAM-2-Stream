"""Lazy model adapter. No global precision changes, GUI calls, or automatic recording."""

from __future__ import annotations

import io
import math
import time
from contextlib import nullcontext

from .config import DetectorConfig
from .contracts import Detection


def normalize_detections(boxes, scores, labels, width: int, height: int) -> list[dict]:
    # Transformers 5.17's tokenizer decodes an empty label batch as [""].
    # It still has zero boxes/scores: a valid no-match result, not a model failure.
    if width > 0 and height > 0 and len(boxes) == len(scores) == 0 and labels in ([], [""]):
        return []
    if width <= 0 or height <= 0 or not (len(boxes) == len(scores) == len(labels)):
        raise ValueError("Invalid detector result dimensions.")
    results = []
    for box, score, label in zip(boxes, scores, labels, strict=True):
        if len(box) != 4 or not all(math.isfinite(float(x)) for x in box):
            continue
        if not isinstance(label, str) or not label.strip():
            continue
        coords = tuple(
            max(0.0, min(1.0, float(x) / scale))
            for x, scale in zip(box, (width, height, width, height), strict=True)
        )
        try:
            result = Detection(label.strip()[:200], float(score), coords)
        except ValueError:
            continue
        results.append(result.json())
    return sorted(results, key=lambda d: d["score"], reverse=True)[:100]


class GroundingDino:
    def __init__(self, config: DetectorConfig):
        import torch
        from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

        self.torch = torch
        self.config = config
        self.device = (
            "cuda" if config.device == "auto" and torch.cuda.is_available() else config.device
        )
        if self.device == "auto":
            self.device = "cpu"
        if self.device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable.")
        kwargs = {"local_files_only": config.local_files_only, "revision": config.revision}
        self.processor = AutoProcessor.from_pretrained(config.model, **kwargs)
        self.model = (
            AutoModelForZeroShotObjectDetection.from_pretrained(config.model, **kwargs)
            .to(self.device)
            .eval()
        )
        self.revision = getattr(self.model.config, "_commit_hash", None) or config.revision
        if self.device == "cuda":
            torch.cuda.reset_peak_memory_stats()

    def detect(self, jpeg: bytes, phrases: tuple[str, ...]) -> dict:
        from PIL import Image

        with Image.open(io.BytesIO(jpeg)) as raw:
            image = raw.convert("RGB")
        width, height = image.size
        started = time.perf_counter()
        resize = {}
        if self.config.image_size is not None:
            resize["size"] = {
                "shortest_edge": self.config.image_size,
                "longest_edge": round(self.config.image_size * 1333 / 800),
            }
        inputs = self.processor(
            images=image, text=". ".join(phrases) + ".", return_tensors="pt", **resize
        ).to(self.device)
        # CUDA fp16 is scoped to this inference; CPU keeps the conservative fp32 path.
        context = (
            self.torch.autocast("cuda", dtype=self.torch.float16)
            if self.device == "cuda"
            else nullcontext()
        )
        with self.torch.inference_mode(), context:
            output = self.model(**inputs)
        result = self.processor.post_process_grounded_object_detection(
            output,
            inputs.input_ids,
            threshold=self.config.threshold,
            text_threshold=self.config.text_threshold,
            target_sizes=[(height, width)],
        )[0]
        labels = result.get("text_labels")
        if labels is None:
            raise RuntimeError("Unsupported Transformers output: text_labels missing.")
        detections = normalize_detections(
            result["boxes"].detach().cpu().tolist(),
            result["scores"].detach().cpu().tolist(),
            labels,
            width,
            height,
        )
        return {
            "detections": detections,
            "width": width,
            "height": height,
            "inference_ms": round((time.perf_counter() - started) * 1000, 2),
            "model": self.config.model,
            "model_revision": self.revision,
            "device": self.device,
            "peak_vram_mb": round(self.torch.cuda.max_memory_allocated() / 1048576, 1)
            if self.device == "cuda"
            else None,
        }


def create_detector(config: DetectorConfig):
    if config.backend == "yoloe":
        from .yoloe import YoloE

        return YoloE(config)
    return GroundingDino(config)
