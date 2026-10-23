"""Optional, offline YOLOE adapter. Only explicitly provisioned local assets are loaded."""

from __future__ import annotations

import hashlib
import io
import os
import time
from pathlib import Path

from .detector import normalize_detections


class YoloE:
    def __init__(self, config):
        # Set before importing Ultralytics: camera workers never install/download dependencies.
        os.environ["YOLO_AUTOINSTALL"] = "false"
        os.environ["YOLO_OFFLINE"] = "true"
        model_path, encoder_path = Path(config.model), Path(config.text_encoder or "")
        if not model_path.is_file() or not encoder_path.is_file():
            raise ValueError("Provision the YOLOE checkpoint and text encoder before starting.")
        import clip  # noqa: F401 -- fail before Ultralytics can try installing it
        import torch
        from ultralytics.nn.text_model import MobileCLIPTS

        from ultralytics import YOLOE

        self.config, self.torch = config, torch
        self.device = config.device
        if self.device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        if self.device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable.")
        self.model = YOLOE(str(model_path)).to(self.device)
        self.model.model.clip_model = MobileCLIPTS(torch.device(self.device), str(encoder_path))
        self.phrases = None
        self.model_name = model_path.name
        with model_path.open("rb") as stream:
            self.revision = "sha256:" + hashlib.file_digest(stream, "sha256").hexdigest()
        with encoder_path.open("rb") as stream:
            self.encoder_revision = "sha256:" + hashlib.file_digest(stream, "sha256").hexdigest()
        if self.device == "cuda":
            torch.cuda.reset_peak_memory_stats()

    def detect(self, jpeg: bytes, phrases: tuple[str, ...]) -> dict:
        from PIL import Image

        started = time.perf_counter()
        with Image.open(io.BytesIO(jpeg)) as raw:
            image = raw.convert("RGB")
        width, height = image.size
        # One cached prompt set, bounded by the request's eight-target limit.
        # Rebuild only when phrases change; never recompute text features per video frame.
        with self.torch.inference_mode():
            if phrases != self.phrases:
                embeddings = self.model.model.get_text_pe(list(phrases), cache_clip_model=True)
                self.model.set_classes(list(phrases), embeddings)
                self.phrases = phrases
            result = self.model.predict(
                image,
                device=self.device,
                imgsz=self.config.image_size or 640,
                conf=self.config.threshold,
                max_det=100,
                verbose=False,
                save=False,
                save_txt=False,
                save_crop=False,
                stream=False,
            )[0]
        boxes = result.boxes
        detections = normalize_detections(
            boxes.xyxy.detach().cpu().tolist(),
            boxes.conf.detach().cpu().tolist(),
            [result.names[int(index)] for index in boxes.cls.detach().cpu().tolist()],
            width,
            height,
        )
        return {
            "detections": detections,
            "width": width,
            "height": height,
            "inference_ms": round((time.perf_counter() - started) * 1000, 2),
            "model": self.model_name,
            "model_revision": self.revision,
            "text_encoder_revision": self.encoder_revision,
            "device": self.device,
            "peak_vram_mb": round(self.torch.cuda.max_memory_allocated() / 1048576, 1)
            if self.device == "cuda"
            else None,
        }
