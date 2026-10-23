"""Compare adapters on identical recorded frames, without opening cameras or saving images."""

import argparse
import io
import json
import statistics
import time
from pathlib import Path

from grounded_stream.config import Config
from grounded_stream.detector import create_detector


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    path = Path(args.output)
    if path.exists():
        parser.error("Choose a new output path.")
    cfg = Config.load(args.config)
    if any(source.kind != "file" for source in cfg.sources):
        parser.error("Use a recorded-file configuration.")
    import cv2
    from PIL import Image

    cap = cv2.VideoCapture(cfg.sources[0].uri)
    frames = []
    try:
        for index in range(12):
            cap.set(cv2.CAP_PROP_POS_MSEC, index * 250)
            ok, frame = cap.read()
            if not ok:
                raise RuntimeError("Fixture needs at least three seconds of video.")
            height, width = frame.shape[:2]
            scale = min(1, 960 / width)
            frame = cv2.resize(frame, (round(width * scale), round(height * scale)))
            ok, encoded = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            if not ok:
                raise RuntimeError("Could not encode fixture.")
            frames.append(encoded.tobytes())
    finally:
        cap.release()
    blank = io.BytesIO()
    Image.new("RGB", (960, 540)).save(blank, format="JPEG")
    started = time.perf_counter()
    detector = create_detector(cfg.detector)
    loaded = time.perf_counter() - started
    cases = []
    for name, phrases, images in [
        ("positive", ("deer",), frames),
        ("negative", ("fire truck",), frames),
        ("blank", ("deer",), [blank.getvalue()] * 12),
        ("multiple", ("deer", "fire truck"), frames),
        ("return_to_positive", ("deer",), frames),
    ]:
        first = detector.detect(images[0], phrases)
        for _ in range(2):
            detector.detect(images[0], phrases)
        results = [detector.detect(image, phrases) for image in images]
        latencies = [result["inference_ms"] for result in results]
        cases.append(
            {
                "case": name,
                "first_frame_ms": first["inference_ms"],
                "warm_p50_ms": statistics.median(latencies),
                "warm_p95_ms": sorted(latencies)[-1],
                "matched_frames": sum(bool(r["detections"]) for r in results),
                "frames": len(results),
                "results": results,
            }
        )
        print(name, cases[-1]["warm_p50_ms"], cases[-1]["matched_frames"], flush=True)
    summary = {
        "load_seconds": loaded,
        "backend": cfg.detector.backend,
        "image_size": cfg.detector.image_size,
        "threshold": cfg.detector.threshold,
        "cases": cases,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(summary, stream, indent=2, allow_nan=False)


if __name__ == "__main__":
    main()
