"""Optional real codec tests; no camera or network is used."""

import multiprocessing as mp
import time

import pytest

from grounded_stream.config import Source
from grounded_stream.runtime import drain, stop_process
from grounded_stream.workers import capture_worker

cv2 = pytest.importorskip("cv2")
np = pytest.importorskip("numpy")


def test_short_file_eof_flushes_final_frame_and_preserves_aspect(tmp_path):
    path = tmp_path / "short.avi"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 10, (640, 480))
    if not writer.isOpened():
        pytest.skip("MJPG writer unavailable")
    for _ in range(6):
        writer.write(np.full((480, 640, 3), 127, dtype=np.uint8))
    writer.release()
    ctx = mp.get_context("spawn")
    frames, statuses, stop = ctx.Queue(1), ctx.Queue(1), ctx.Event()
    source = Source("file", "Video", "file", str(path), 5, 320)
    process = ctx.Process(target=capture_worker, args=(source, frames, statuses, stop))
    received = []
    states = []
    process.start()
    try:
        deadline = time.monotonic() + 10
        while process.is_alive() and time.monotonic() < deadline:
            frame = drain(frames)
            status = drain(statuses)
            if frame:
                received.append(frame)
            if status:
                states.append(status["state"])
            time.sleep(0.01)
        frame = drain(frames)
        status = drain(statuses)
        if frame:
            received.append(frame)
        if status:
            states.append(status["state"])
        assert not process.is_alive()
        assert received
        assert states[-1] == "ended"
        image = cv2.imdecode(np.frombuffer(received[-1]["jpeg"], dtype=np.uint8), cv2.IMREAD_COLOR)
        assert image.shape[:2] == (240, 320)
        assert received[-1]["original_width"] == 640
    finally:
        stop.set()
        stop_process(process)
        for channel in (frames, statuses):
            channel.cancel_join_thread()
            channel.close()
