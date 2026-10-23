"""Spawn-safe process boundaries for blocking native camera and model operations."""

from __future__ import annotations

import os
import queue
import time
import traceback

from .contracts import utc_now

MAX_JPEG_BYTES = 1_500_000


def latest_put(channel, value):
    """A one-item mailbox: replace obsolete data, never wait behind a slow consumer."""
    try:
        channel.put_nowait(value)
        return
    except queue.Full:
        pass
    try:
        channel.get_nowait()
    except queue.Empty:
        return
    try:
        channel.put_nowait(value)
    except queue.Full:
        pass


def capture_worker(source, frames, statuses, stop):
    import cv2

    # If the supervisor terminates a native read, no queued frame must delay shutdown.
    cap = None
    sequence = 0
    backoff = 0.25
    started = time.monotonic()
    try:
        while not stop.is_set():
            cap = cv2.VideoCapture()
            if source.kind == "mjpeg":
                opened = cap.open(
                    source.uri,
                    cv2.CAP_FFMPEG,
                    [cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 3000, cv2.CAP_PROP_READ_TIMEOUT_MSEC, 3000],
                )
            else:
                opened = cap.open(source.uri)
            if not opened:
                latest_put(statuses, {"state": "unavailable", "detail": "Could not open source."})
                cap.release()
                if source.kind == "file":
                    return
                if stop.wait(backoff):
                    return
                backoff = min(backoff * 2, 5)
                continue
            latest_put(statuses, {"state": "running"})
            backoff = 0.25
            last_sample = -float("inf")
            fps = cap.get(cv2.CAP_PROP_FPS) if source.kind == "file" else 0
            if not 0 < fps <= 240:
                fps = 30
            file_index = 0
            play_started = time.monotonic()
            while not stop.is_set():
                ok, frame = cap.read()
                if not ok:
                    latest_put(
                        statuses, {"state": "ended" if source.kind == "file" else "disconnected"}
                    )
                    break
                file_index += 1
                # File playback is paced, rather than decoding the entire file before inference starts.
                media_ms = (file_index - 1) * 1000 / fps if source.kind == "file" else None
                if media_ms is not None:
                    if stop.wait(max(0, play_started + media_ms / 1000 - time.monotonic())):
                        return
                now = time.monotonic()
                if now - last_sample < 1 / source.sample_fps:
                    continue
                last_sample = now
                height, width = frame.shape[:2]
                scale = min(1, source.max_width / width, 1080 / height)
                if scale < 1:
                    frame = cv2.resize(
                        frame, (max(1, round(width * scale)), max(1, round(height * scale)))
                    )
                ok, encoded = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
                if not ok or encoded.nbytes > MAX_JPEG_BYTES:
                    latest_put(
                        statuses,
                        {"state": "error", "detail": "Frame encoding failed or exceeded limit."},
                    )
                    continue
                sequence += 1
                latest_put(
                    frames,
                    {
                        "source_id": source.id,
                        "frame_id": f"{source.id}:{sequence}",
                        "captured_at": utc_now(),
                        "captured_mono": time.monotonic(),
                        "media_time_ms": media_ms,
                        "original_width": width,
                        "original_height": height,
                        "jpeg": encoded.tobytes(),
                        "capture_elapsed_ms": round((now - started) * 1000, 1),
                    },
                )
            cap.release()
            cap = None
            if source.kind == "file":
                return
            stop.wait(backoff)
    except Exception:
        latest_put(statuses, {"state": "error", "detail": "Capture worker failed."})
    finally:
        if cap is not None:
            cap.release()


def detector_worker(config, commands, replies):
    try:
        from .detector import create_detector

        detector = create_detector(config)
        replies.put({"kind": "ready", "device": detector.device})
        while True:
            command = commands.get()
            if command is None:
                return
            request_id, frame, phrases = command
            try:
                result = detector.detect(frame["jpeg"], phrases)
                replies.put({"kind": "result", "request_id": request_id, "value": result})
            except Exception as exc:
                if os.environ.get("GROUNDED_STREAM_DEBUG") == "1":
                    traceback.print_exc()
                # Do not include user prompts, paths, or camera addresses in errors.
                replies.put(
                    {
                        "kind": "error",
                        "request_id": request_id,
                        "error": f"Inference failed ({type(exc).__name__}).",
                    }
                )
    except Exception as exc:
        if os.environ.get("GROUNDED_STREAM_DEBUG") == "1":
            traceback.print_exc()
        replies.put(
            {
                "kind": "error",
                "error": f"Detector could not start ({type(exc).__name__}). Check installed vision extras, model cache, and device.",
            }
        )
