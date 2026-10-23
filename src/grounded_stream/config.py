"""Trusted operator configuration. API requests can reference IDs, never arbitrary paths."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from urllib.parse import urlsplit

from .contracts import finite, identifier


@dataclass(frozen=True)
class Source:
    id: str
    name: str
    kind: str
    uri: str | int
    sample_fps: float = 1
    max_width: int = 960

    @classmethod
    def parse(cls, raw: dict, base: Path) -> Source:
        if not isinstance(raw, dict) or set(raw) - {
            "id",
            "name",
            "kind",
            "uri",
            "sample_fps",
            "max_width",
        }:
            raise ValueError("Unsupported source configuration.")
        sid = identifier(raw.get("id"))
        name = raw.get("name", sid)
        if not isinstance(name, str) or not 1 <= len(name.strip()) <= 100:
            raise ValueError("Invalid source name.")
        kind, uri = raw.get("kind"), raw.get("uri")
        if kind == "file":
            if not isinstance(uri, str) or not uri:
                raise ValueError("File sources need a path.")
            path = Path(uri)
            uri = str((base / path).resolve() if not path.is_absolute() else path.resolve())
            if not Path(uri).is_file():
                raise ValueError(f"Source {sid}: video file does not exist.")
        elif kind == "webcam":
            if isinstance(uri, bool) or not isinstance(uri, int) or not 0 <= uri <= 32:
                raise ValueError("Webcam index must be an integer from 0 to 32.")
        elif kind == "mjpeg":
            if not isinstance(uri, str):
                raise ValueError("MJPEG source needs an HTTP(S) URL.")
            u = urlsplit(uri)
            if u.scheme not in {"http", "https"} or not u.hostname or u.username or u.password:
                raise ValueError("MJPEG URL must use HTTP(S), without embedded credentials.")
            if u.port is not None and not 1 <= u.port <= 65535:
                raise ValueError("Invalid MJPEG port.")
            if u.fragment or u.query:
                raise ValueError("MJPEG URL query strings and fragments are not supported.")
        else:
            raise ValueError("Source kind must be file, webcam, or mjpeg.")
        fps = finite(raw.get("sample_fps", 1), 0.2, 5, "sample_fps")
        width = raw.get("max_width", 960)
        if isinstance(width, bool) or not isinstance(width, int) or not 160 <= width <= 1920:
            raise ValueError("max_width must be an integer from 160 to 1920.")
        return cls(sid, name.strip(), kind, uri, fps, width)

    def public(self) -> dict:
        return {k: v for k, v in asdict(self).items() if k != "uri"}


@dataclass(frozen=True)
class DetectorConfig:
    model: str = "IDEA-Research/grounding-dino-tiny"
    revision: str | None = None
    device: str = "auto"
    threshold: float = 0.35
    text_threshold: float = 0.25
    local_files_only: bool = False
    backend: str = "grounding_dino"
    text_encoder: str | None = None
    image_size: int | None = None

    @classmethod
    def parse(cls, raw: dict) -> DetectorConfig:
        if not isinstance(raw, dict) or set(raw) - set(cls.__dataclass_fields__):
            raise ValueError("Unsupported detector configuration.")
        cfg = cls(**raw)
        if not isinstance(cfg.backend, str) or cfg.backend not in {"grounding_dino", "yoloe"}:
            raise ValueError("backend must be grounding_dino or yoloe.")
        if cfg.image_size is not None and (
            isinstance(cfg.image_size, bool)
            or not isinstance(cfg.image_size, int)
            or not 320 <= cfg.image_size <= 1280
            or cfg.image_size % 32
        ):
            raise ValueError("image_size must be a multiple of 32 between 320 and 1280.")
        if cfg.backend == "yoloe":
            if (
                not isinstance(cfg.model, str)
                or not cfg.model.endswith(".pt")
                or "-pf." in cfg.model
            ):
                raise ValueError(
                    "YOLOE needs a local text-prompted .pt checkpoint, not a prompt-free model."
                )
            if not isinstance(cfg.text_encoder, str) or not cfg.text_encoder.endswith(".ts"):
                raise ValueError("YOLOE requires a local TorchScript text_encoder path.")
            if cfg.revision is not None:
                raise ValueError("YOLOE identifies local weights by SHA-256, not a hub revision.")
        elif cfg.text_encoder is not None:
            raise ValueError("text_encoder is only supported by YOLOE.")
        if not isinstance(cfg.device, str) or cfg.device not in {"auto", "cpu", "cuda"}:
            raise ValueError("device must be auto, cpu, or cuda.")
        if not isinstance(cfg.model, str) or not 1 <= len(cfg.model) <= 200:
            raise ValueError("Invalid model identifier.")
        if cfg.revision is not None and (
            not isinstance(cfg.revision, str) or not 1 <= len(cfg.revision) <= 100
        ):
            raise ValueError("Invalid model revision.")
        if not isinstance(cfg.local_files_only, bool):
            raise ValueError("local_files_only must be boolean.")
        finite(cfg.threshold, 0, 1, "threshold")
        finite(cfg.text_threshold, 0, 1, "text_threshold")
        return cfg


@dataclass(frozen=True)
class Config:
    sources: tuple[Source, ...]
    detector: DetectorConfig = DetectorConfig()

    @classmethod
    def load(cls, path: str | Path) -> Config:
        path = Path(path).resolve()
        if path.stat().st_size > 65536:
            raise ValueError("Configuration exceeds 64 KiB.")
        raw = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(raw, dict) or set(raw) - {"sources", "detector"}:
            raise ValueError("Unsupported configuration fields.")
        sources = raw.get("sources")
        if not isinstance(sources, list) or not 1 <= len(sources) <= 8:
            raise ValueError("Configure between one and eight sources.")
        parsed = tuple(Source.parse(s, path.parent) for s in sources)
        if len({s.id for s in parsed}) != len(parsed):
            raise ValueError("Source IDs must be unique.")
        detector = DetectorConfig.parse(raw.get("detector", {}))
        if detector.backend == "yoloe":
            from dataclasses import replace

            paths = {}
            for name in ("model", "text_encoder"):
                asset = Path(getattr(detector, name))
                asset = (
                    (path.parent / asset).resolve() if not asset.is_absolute() else asset.resolve()
                )
                if not asset.is_file():
                    raise ValueError(
                        f"YOLOE {name} file is missing; provision assets before starting."
                    )
                paths[name] = str(asset)
            detector = replace(detector, **paths)
        return cls(parsed, detector)
