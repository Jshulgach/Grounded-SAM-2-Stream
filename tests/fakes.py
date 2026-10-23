"""Spawn-safe workers: no camera, model download, or GPU required."""

import time

from grounded_stream.contracts import utc_now
from grounded_stream.workers import latest_put


def capture(source, frames, statuses, stop):
    if source.id == "blocked":
        stop.wait(60)
        return
    if source.id == "bad":
        statuses.put({"state": "unavailable"})
        return
    statuses.put({"state": "running"})
    for n in range(10000):
        if stop.is_set():
            return
        latest_put(
            frames,
            {
                "source_id": source.id,
                "frame_id": f"{source.id}:{n}",
                "captured_at": utc_now(),
                "captured_mono": time.monotonic(),
                "jpeg": b"jpeg" * 256,
                "original_width": 320,
                "original_height": 180,
            },
        )
        if source.id == "eof" and n == 2:
            statuses.put({"state": "ended"})
            return
        if source.id == "stalled":
            stop.wait(60)
            return
        stop.wait(0.04)


def detector(config, commands, replies):
    if config.model == "blocked-load":
        time.sleep(60)
        return
    if config.model == "failed-load":
        replies.put({"kind": "error", "error": "Test startup failure."})
        return
    replies.put({"kind": "ready"})
    while True:
        command = commands.get()
        if command is None:
            return
        jid, frame, targets = command
        if targets[0] == "blocked-inference":
            time.sleep(60)
        if targets[0] == "slow":
            time.sleep(0.12)
        replies.put(
            {
                "kind": "result",
                "request_id": jid,
                "value": {
                    "detections": [{"label": targets[0], "score": 0.8, "box": [0, 0, 0.5, 0.5]}],
                    "width": 320,
                    "height": 180,
                    "inference_ms": 120 if targets[0] == "slow" else 1,
                    "model": "test-double",
                    "model_revision": "fixture",
                    "device": "test",
                },
            }
        )
