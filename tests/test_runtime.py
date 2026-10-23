import time
from dataclasses import replace

import pytest

from grounded_stream.config import Config, DetectorConfig, Source
from grounded_stream.contracts import TERMINAL, Search
from grounded_stream.runtime import BusyError


def config(*ids, model="test"):
    return Config(
        tuple(Source(sid, sid, "file" if sid == "eof" else "webcam", 0) for sid in ids),
        DetectorConfig(model=model),
    )


def submit(manager, target="toy", duration=1, timeout=15):
    return manager.submit(
        Search((target,), tuple(s.id for s in manager.config.sources), duration, timeout)
    )["id"]


def wait(manager, jid, predicate=lambda j: j["status"] in TERMINAL, timeout=20):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        job = manager.snapshot(jid)
        if predicate(job):
            return job
        time.sleep(0.02)
    raise AssertionError(manager.snapshot(jid))


def test_multisource_failure_does_not_starve_good_source(managers):
    manager = managers("bad", "eof", "good")
    job = wait(manager, submit(manager, duration=2))
    assert job["status"] == "partial"
    assert job["sources"]["bad"]["processed_frames"] == 0
    assert job["sources"]["eof"]["processed_frames"] > 0
    assert job["sources"]["good"]["processed_frames"] > 0


def test_completed_no_false_recording(managers):
    manager = managers("good")
    job = wait(manager, submit(manager))
    assert job["status"] == "completed"
    obs = job["observations"][-1]
    jpeg, frame_id = manager.evidence(job["id"], "good")
    assert jpeg and frame_id == obs["frame_id"]
    assert frame_id in obs["evidence_url"]
    assert "jpeg" not in obs
    assert job["metrics"]["inference_p50_ms"] == 1


@pytest.mark.parametrize("model,target", [("blocked-load", "toy"), ("test", "blocked-inference")])
def test_cancellation_kills_blocked_native_work_and_recovers(managers, model, target):
    manager = managers("good", model=model)
    jid = submit(manager, target=target, duration=5)
    wait(
        manager, jid, lambda j: j["status"] == ("loading" if model == "blocked-load" else "running")
    )
    start = time.perf_counter()
    assert manager.cancel(jid)["status"] == "cancelled"
    assert time.perf_counter() - start < 0.5
    deadline = time.monotonic() + 5
    while manager.health()["active_search"] is not None and time.monotonic() < deadline:
        time.sleep(0.02)
    assert manager.health()["active_search"] is None
    if model == "blocked-load":
        manager.config = replace(manager.config, detector=DetectorConfig(model="test"))
    assert wait(manager, submit(manager))["status"] == "completed"


def test_initialization_failure_is_reported(managers):
    manager = managers("good", model="failed-load")
    job = wait(manager, submit(manager))
    assert job["status"] == "failed"
    assert "startup failure" in job["error"]


def test_deadline_includes_model_startup(managers):
    manager = managers("good", model="blocked-load")
    start = time.perf_counter()
    assert wait(manager, submit(manager, timeout=0.4))["status"] == "timed_out"
    assert time.perf_counter() - start < 3


def test_queue_capacity_and_cancelled_retention_race(managers):
    manager = managers("good", model="blocked-load", max_pending=2, max_jobs=4, retention_seconds=0)
    active = submit(manager, timeout=15)
    wait(manager, active, lambda j: j["status"] == "loading")
    a, b = submit(manager), submit(manager)
    with pytest.raises(BusyError):
        submit(manager)
    manager.cancel(a)
    manager.cancel(b)
    for _ in range(12):
        jid = submit(manager)
        manager.cancel(jid)
    manager.cancel(active)
    deadline = time.monotonic() + 5
    while manager.active is not None and time.monotonic() < deadline:
        time.sleep(0.02)
    assert manager.thread.is_alive()
    assert manager.health()["retained_searches"] <= 4


def test_evidence_budget_and_observation_window(managers):
    manager = managers("a", "b", evidence_budget=1100)
    job = wait(manager, submit(manager, duration=2))
    assert job["processed_frames"] > 5
    assert len(job["observations"]) <= 128
    assert manager.health()["evidence_bytes"] <= 1100


def test_slow_inference_uses_recent_frames_and_both_sources(managers):
    manager = managers("a", "b")
    job = wait(manager, submit(manager, target="slow", duration=2))
    assert all(s["processed_frames"] > 1 for s in job["sources"].values())
    assert max(o["frame_age_ms"] for o in job["observations"]) < 1500


def test_blocked_capture_is_bounded_and_does_not_block_health(managers):
    manager = managers("blocked", "good")
    jid = submit(manager, duration=1)
    start = time.perf_counter()
    manager.health()
    assert time.perf_counter() - start < 0.2
    job = wait(manager, jid)
    assert job["status"] == "partial"


def test_stalled_source_not_reported_healthy(managers):
    manager = managers("stalled")
    job = wait(manager, submit(manager, duration=6))
    assert job["status"] == "failed"
    assert job["sources"]["stalled"]["state"] == "stalled"


def test_cancel_is_idempotent_and_late_results_are_ignored(managers):
    manager = managers("good")
    jid = submit(manager, target="slow", duration=5)
    wait(manager, jid, lambda j: j["processed_frames"] > 0)
    cancelled = manager.cancel(jid)
    time.sleep(0.4)
    assert manager.cancel(jid) == cancelled
