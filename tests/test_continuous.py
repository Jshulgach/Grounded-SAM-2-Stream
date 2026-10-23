import time

import pytest

from grounded_stream.contracts import Search
from tests.test_runtime import submit, wait


def test_continuous_contract_is_explicit_and_old_default_remains_timed():
    body = {"targets": ["toy"], "source_ids": ["good"]}
    assert Search.parse(body, {"good"}).duration_seconds == 10
    assert Search.parse({**body, "duration_seconds": None}, {"good"}).duration_seconds is None


@pytest.mark.parametrize("timeout", [None, True, float("inf")])
def test_continuous_still_requires_bounded_operation_timeout(timeout):
    with pytest.raises(ValueError):
        Search.parse(
            {
                "targets": ["toy"],
                "source_ids": ["good"],
                "duration_seconds": None,
                "timeout_seconds": timeout,
            },
            {"good"},
        )


def test_continuous_survives_old_deadline_and_keeps_memory_bounded(managers):
    manager = managers("good", evidence_budget=2048)
    jid = submit(manager, duration=None, timeout=3)
    job = wait(manager, jid, lambda j: j["processed_frames"] >= 150, timeout=15)
    assert job["status"] == "running"
    assert job["metrics"]["elapsed_ms"] > 3000
    assert len(job["observations"]) == 128
    assert job["metrics"]["latency_sample_count"] == 128
    assert manager.health()["evidence_bytes"] <= 2048
    manager.cancel(jid)
    assert wait(manager, submit(manager))["status"] == "completed"


def test_continuous_startup_can_time_out(managers):
    manager = managers("good", model="blocked-load")
    job = wait(manager, submit(manager, duration=None, timeout=0.3))
    assert job["status"] == "timed_out"


def test_continuous_inference_cannot_hang_forever(managers):
    manager = managers("good")
    job = wait(manager, submit(manager, target="blocked-inference", duration=None, timeout=2))
    assert job["status"] == "timed_out"
    assert job["error"] == "Inference deadline exceeded."
    assert wait(manager, submit(manager))["status"] == "completed"


def test_queued_search_expires_while_continuous_search_stays_active(managers):
    manager = managers("good")
    jid = submit(manager, duration=None)
    wait(manager, jid, lambda j: j["processed_frames"] > 0)
    queued = submit(manager, timeout=0.2)
    job = wait(manager, queued)
    assert job["status"] == "timed_out"
    assert "queue" in job["error"]
    assert manager.snapshot(jid)["status"] == "running"
    manager.cancel(jid)


def test_continuous_files_finish_at_eof(managers):
    manager = managers("eof")
    job = wait(manager, submit(manager, duration=None))
    assert job["status"] == "completed"
    assert job["sources"]["eof"]["state"] == "ended"


def test_continuous_stalled_camera_is_visible_and_cancellable(managers):
    manager = managers("stalled", "good")
    jid = submit(manager, duration=None)
    job = wait(manager, jid, lambda j: j["sources"]["stalled"]["state"] == "stalled", timeout=10)
    assert job["status"] == "running"
    assert job["sources"]["good"]["processed_frames"] > 0
    start = time.perf_counter()
    assert manager.cancel(jid)["status"] == "cancelled"
    assert time.perf_counter() - start < 0.5
