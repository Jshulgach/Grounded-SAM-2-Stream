import time

import pytest

from grounded_stream.contracts import utc_now
from tests.test_runtime import submit, wait


def test_result_window_and_memory_remain_bounded_under_burst(managers):
    manager = managers("good", model="blocked-load", evidence_budget=4096)
    jid = submit(manager, timeout=30)
    wait(manager, jid, lambda job: job["status"] == "loading")
    with manager.lock:
        for n in range(1000):
            manager._record(
                jid,
                {
                    "source_id": "good",
                    "frame_id": str(n),
                    "jpeg": b"x" * 1024,
                    "captured_mono": time.monotonic(),
                    "captured_at": utc_now(),
                },
                {"inference_ms": 1, "detections": []},
            )
    job = manager.snapshot(jid)
    assert job["processed_frames"] == 1000
    assert len(job["observations"]) == 128
    assert job["observations"][0]["frame_id"] == "872"
    assert manager.health()["evidence_bytes"] == 1024
    assert len(manager.jobs[jid]["_latencies"]) == 128
    manager.cancel(jid)


def test_completed_results_expire_without_new_searches(managers):
    manager = managers("good", retention_seconds=0.1)
    jid = submit(manager)
    wait(manager, jid)
    deadline = time.monotonic() + 3
    while jid in manager.jobs and time.monotonic() < deadline:
        time.sleep(0.02)
    with pytest.raises(KeyError):
        manager.snapshot(jid)
    assert manager.health()["evidence_bytes"] == 0
