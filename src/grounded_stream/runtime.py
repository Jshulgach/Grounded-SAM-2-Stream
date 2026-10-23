"""Finite or continuous, memory-bounded search sessions with a responsive supervisor and killable native workers."""

from __future__ import annotations

import copy
import multiprocessing as mp
import queue
import statistics
import threading
import time
import uuid
from collections import deque
from dataclasses import asdict

from .config import Config
from .contracts import SCHEMA_VERSION, TERMINAL, Search, utc_now
from .workers import capture_worker, detector_worker


class BusyError(RuntimeError):
    pass


def stop_process(process):
    if process is None or process.pid is None:
        return
    if process.is_alive():
        process.terminate()
    process.join(0.5)
    if process.is_alive():
        process.kill()
        process.join(0.5)
    if process.is_alive():
        raise RuntimeError("Worker did not stop.")
    process.close()


def drain(channel):
    value = None
    while True:
        try:
            value = channel.get_nowait()
        except queue.Empty:
            return value


class SearchManager:
    def __init__(
        self,
        config: Config,
        *,
        capture_target=capture_worker,
        detector_target=detector_worker,
        max_pending=4,
        max_jobs=16,
        retention_seconds=900,
        evidence_budget=32 * 1024 * 1024,
    ):
        self.config = config
        self.capture_target = capture_target
        self.detector_target = detector_target
        self.max_pending = max_pending
        self.max_jobs = max_jobs
        self.retention_seconds = retention_seconds
        self.evidence_budget = evidence_budget
        self.ctx = mp.get_context("spawn")
        self.lock = threading.RLock()
        self.wake = threading.Event()
        self.stopping = threading.Event()
        self.pending = deque()
        self.jobs = {}
        self.active = None
        self.thread = None
        self.detector = None
        self.commands = None
        self.replies = None
        self.detector_state = "unloaded"

    def start(self):
        if self.thread is not None:
            raise RuntimeError("Manager already started.")
        self.thread = threading.Thread(target=self._run, name="search-supervisor", daemon=True)
        self.thread.start()

    def _prune(self, make_room=True):
        now = time.monotonic()
        for jid, job in list(self.jobs.items()):
            if (
                jid != self.active
                and job["status"] in TERMINAL
                and now - job["_finished"] > self.retention_seconds
            ):
                del self.jobs[jid]
        while make_room and len(self.jobs) >= self.max_jobs:
            old = next(
                (k for k, j in self.jobs.items() if k != self.active and j["status"] in TERMINAL),
                None,
            )
            if old is None:
                raise BusyError("Search history is full. Wait for an active search to finish.")
            del self.jobs[old]

    def submit(self, request: Search) -> dict:
        with self.lock:
            if self.stopping.is_set() or self.thread is None or not self.thread.is_alive():
                raise BusyError("Search service is stopping or unavailable.")
            self._prune()
            self.pending = deque(
                k for k in self.pending if k in self.jobs and self.jobs[k]["status"] == "queued"
            )
            if len(self.pending) >= self.max_pending:
                raise BusyError("Search queue is full. Cancel a search or try again later.")
            jid = uuid.uuid4().hex
            self.jobs[jid] = {
                "schema_version": SCHEMA_VERSION,
                "id": jid,
                "status": "queued",
                "request": asdict(request),
                "created_at": utc_now(),
                "finished_at": None,
                "sources": {
                    sid: {"state": "waiting", "processed_frames": 0} for sid in request.source_ids
                },
                "observations": [],
                "processed_frames": 0,
                "error": None,
                "metrics": {},
                "_request": request,
                "_started": time.monotonic(),
                "_finished": 0,
                "_evidence": {},
                "_latencies": [],
            }
            self.pending.append(jid)
            self.wake.set()
            return self.snapshot(jid)

    def snapshot(self, jid: str, *, latest_only: bool = False) -> dict:
        with self.lock:
            job = self.jobs[jid]
            public = {k: v for k, v in job.items() if not k.startswith("_")}
            if latest_only:
                # Select before copying: preview polling should not serialize rolling history.
                latest = {item["source_id"]: item for item in job["observations"]}
                public["observations"] = list(latest.values())
            result = copy.deepcopy(public)
            if job["status"] not in TERMINAL:
                latencies = sorted(job["_latencies"])
                result["metrics"] = {
                    "elapsed_ms": round((time.monotonic() - job["_started"]) * 1000, 1),
                    "inference_p50_ms": round(statistics.median(latencies), 2)
                    if latencies
                    else None,
                    "latency_sample_count": len(latencies),
                }
            return result

    def evidence(self, jid: str, sid: str) -> tuple[bytes, str]:
        with self.lock:
            value = self.jobs[jid]["_evidence"][sid]
            return value["jpeg"], value["frame_id"]

    def cancel(self, jid: str) -> dict:
        with self.lock:
            if self.jobs[jid]["status"] not in TERMINAL:
                self._finish(jid, "cancelled")
                self.wake.set()
            return self.snapshot(jid)

    def _finish(self, jid: str, status: str, error=None):
        job = self.jobs[jid]
        if job["status"] in TERMINAL:
            return
        job.update(status=status, error=error, finished_at=utc_now(), _finished=time.monotonic())
        for source in job["sources"].values():
            if source["state"] == "running":
                source["state"] = "stopped"
            elif source["state"] == "waiting":
                source["state"] = "not_started"
        latencies = sorted(job["_latencies"])
        job["metrics"] = {
            "elapsed_ms": round((job["_finished"] - job["_started"]) * 1000, 1),
            "inference_p50_ms": round(statistics.median(latencies), 2) if latencies else None,
            "inference_p95_ms": latencies[min(len(latencies) - 1, int(len(latencies) * 0.95))]
            if latencies
            else None,
            "latency_sample_count": len(latencies),
        }

    def health(self) -> dict:
        with self.lock:
            return {
                "schema_version": SCHEMA_VERSION,
                "status": (
                    "stopping"
                    if self.stopping.is_set()
                    else "ready"
                    if self.thread and self.thread.is_alive()
                    else "unavailable"
                ),
                "detector": self.detector_state,
                "active_search": self.active,
                "queued": sum(j["status"] == "queued" for j in self.jobs.values()),
                "retained_searches": len(self.jobs),
                "evidence_bytes": sum(
                    len(e["jpeg"]) for j in self.jobs.values() for e in j["_evidence"].values()
                ),
                "sources": [s.public() for s in self.config.sources],
            }

    def close(self):
        self.stopping.set()
        with self.lock:
            for jid in self.jobs:
                self._finish(jid, "cancelled")
        self.wake.set()
        if self.thread:
            self.thread.join(15)
            if self.thread.is_alive():
                raise RuntimeError("Search supervisor did not stop.")

    def _interrupted(self, jid):
        with self.lock:
            now = time.monotonic()
            # A continuous active session must not keep expired queued jobs alive.
            for pending_id in list(self.pending):
                pending_job = self.jobs.get(pending_id)
                if (
                    pending_job
                    and pending_job["status"] == "queued"
                    and (now - pending_job["_started"] >= pending_job["_request"].timeout_seconds)
                ):
                    self._finish(pending_id, "timed_out", "Search expired in the queue.")
            self._prune(make_room=False)
            job = self.jobs[jid]
            if self.stopping.is_set() or job["status"] in TERMINAL:
                return True
            request = job["_request"]
            if request.duration_seconds is None and job["status"] == "running":
                return False
            if now - job["_started"] >= request.timeout_seconds:
                self._finish(jid, "timed_out", "Search deadline exceeded.")
                return True
            return False

    def _reset_detector(self):
        stop_process(self.detector)
        self.detector = None
        for channel in (self.commands, self.replies):
            if channel is not None:
                channel.cancel_join_thread()
                channel.close()
        self.commands = self.replies = None
        with self.lock:
            self.detector_state = "unloaded"

    def _ready_detector(self, jid):
        if self.detector is not None and self.detector.is_alive():
            return True
        self._reset_detector()
        self.commands = self.ctx.Queue(maxsize=1)
        self.replies = self.ctx.Queue(maxsize=1)
        self.detector = self.ctx.Process(
            target=self.detector_target,
            args=(self.config.detector, self.commands, self.replies),
            name="grounded-detector",
            daemon=True,
        )
        with self.lock:
            self.detector_state = "loading"
        self.detector.start()
        while not self._interrupted(jid):
            reply = drain(self.replies)
            if reply:
                if reply["kind"] == "ready":
                    with self.lock:
                        self.detector_state = "ready"
                    return True
                raise RuntimeError(reply.get("error", "Detector initialization failed."))
            if not self.detector.is_alive():
                raise RuntimeError("Detector worker exited during initialization.")
            self.stopping.wait(0.02)
        self._reset_detector()
        return False

    def _infer(self, jid, frame, request):
        self.commands.put_nowait((jid, frame, request.targets))
        inference_deadline = time.monotonic() + request.timeout_seconds
        while not self._interrupted(jid):
            if time.monotonic() >= inference_deadline:
                with self.lock:
                    self._finish(jid, "timed_out", "Inference deadline exceeded.")
                break
            reply = drain(self.replies)
            if reply:
                if reply.get("request_id") != jid:
                    continue
                if reply["kind"] == "error":
                    raise RuntimeError(reply["error"])
                return reply["value"]
            if not self.detector.is_alive():
                raise RuntimeError("Detector worker exited.")
            self.stopping.wait(0.02)
        # A cancelled CUDA/native call cannot be stopped by cancelling an asyncio task.
        self._reset_detector()
        return None

    def _record(self, jid, frame, result):
        with self.lock:
            job = self.jobs[jid]
            if job["status"] in TERMINAL:
                return
            source_id = frame["source_id"]
            observation = {k: v for k, v in frame.items() if k not in {"jpeg", "captured_mono"}}
            observation.update(
                result,
                processed_at=utc_now(),
                frame_age_ms=round((time.monotonic() - frame["captured_mono"]) * 1000, 1),
            )
            observation["evidence_url"] = (
                f"/v1/searches/{jid}/evidence/{source_id}?frame_id={frame['frame_id']}"
            )
            job["observations"].append(observation)
            job["observations"] = job["observations"][-128:]
            job["_evidence"][source_id] = {"jpeg": frame["jpeg"], "frame_id": frame["frame_id"]}
            job["processed_frames"] += 1
            job["sources"][source_id]["processed_frames"] += 1
            job["_latencies"] = (job["_latencies"] + [result["inference_ms"]])[-128:]
            # Drop oldest previews first. Metadata remains available.
            total = sum(len(e["jpeg"]) for j in self.jobs.values() for e in j["_evidence"].values())
            for candidate in self.jobs.values():
                if total <= self.evidence_budget:
                    break
                for sid in list(candidate["_evidence"]):
                    total -= len(candidate["_evidence"].pop(sid)["jpeg"])
                    if total <= self.evidence_budget:
                        break

    def _search(self, jid):
        with self.lock:
            if self._interrupted(jid):
                return
            request = self.jobs[jid]["_request"]
            self.jobs[jid]["status"] = "loading"
        if not self._ready_detector(jid):
            return
        captures = []
        capture_timer = None
        stop = self.ctx.Event()
        try:
            for source in self.config.sources:
                if source.id not in request.source_ids:
                    continue
                frames, statuses = self.ctx.Queue(maxsize=1), self.ctx.Queue(maxsize=1)
                process = self.ctx.Process(
                    target=self.capture_target,
                    args=(source, frames, statuses, stop),
                    name=f"capture-{source.id}",
                    daemon=True,
                )
                process.start()
                captures.append((source, process, frames, statuses))
            with self.lock:
                if self.jobs[jid]["status"] not in TERMINAL:
                    self.jobs[jid]["status"] = "running"
            # Duration measures capture time; the overall deadline includes model startup/queueing.
            capture_started = time.monotonic()
            end = float("inf")
            if request.duration_seconds is not None:
                end = capture_started + request.duration_seconds
                capture_timer = threading.Timer(request.duration_seconds, stop.set)
                capture_timer.daemon = True
                capture_timer.start()
            seen = {}
            latest = {}
            while time.monotonic() < end and not self._interrupted(jid):
                worked = False
                for source, process, frames, statuses in captures:
                    if time.monotonic() >= end or self._interrupted(jid):
                        break
                    status = drain(statuses)
                    frame = drain(frames)
                    with self.lock:
                        if self._interrupted(jid):
                            return
                        if frame:
                            latest[source.id] = frame
                            self.jobs[jid]["sources"][source.id].update(
                                state="running", last_frame_at=frame["captured_at"]
                            )
                            self.jobs[jid]["sources"][source.id].pop("detail", None)
                        if status:
                            self.jobs[jid]["sources"][source.id].update(status)
                    frame = latest.get(source.id)
                    if (
                        source.kind != "file"
                        and time.monotonic()
                        - (frame["captured_mono"] if frame else capture_started)
                        > 5
                    ):
                        with self.lock:
                            state = self.jobs[jid]["sources"][source.id]
                            if state["state"] in {"running", "waiting"}:
                                state.update(state="stalled", detail="Waiting for fresh frames.")
                    if frame and frame["frame_id"] != seen.get(source.id):
                        seen[source.id] = frame["frame_id"]
                        # Do not infer a frame already obsolete from a stalled live source.
                        if time.monotonic() - frame["captured_mono"] > 5:
                            continue
                        result = self._infer(jid, frame, request)
                        if result is None:
                            return
                        self._record(jid, frame, result)
                        worked = True
                    elif not process.is_alive():
                        with self.lock:
                            state = self.jobs[jid]["sources"][source.id]["state"]
                            if state in {"running", "waiting"}:
                                self.jobs[jid]["sources"][source.id]["state"] = "error"
                if all(not p.is_alive() for _, p, _, _ in captures) and not worked:
                    break
                if not worked:
                    self.stopping.wait(0.02)
            with self.lock:
                if self.jobs[jid]["status"] in TERMINAL:
                    return
                for source, _process, frames, statuses in captures:
                    status = drain(statuses)
                    state = self.jobs[jid]["sources"][source.id]
                    if status:
                        state.update(status)
                    pending_frame = drain(frames)
                    frame = pending_frame or latest.get(source.id)
                    if (
                        source.kind != "file"
                        and state["state"] == "running"
                        and (
                            frame is None or min(time.monotonic(), end) - frame["captured_mono"] > 5
                        )
                    ):
                        state["state"] = "stalled"
                sources = self.jobs[jid]["sources"].values()
                healthy = [
                    s["processed_frames"] > 0 and s["state"] in {"running", "ended"}
                    for s in sources
                ]
                self._finish(
                    jid,
                    "completed" if all(healthy) else "partial" if any(healthy) else "failed",
                    None if all(healthy) else "Some sources did not provide usable frames.",
                )
        finally:
            if capture_timer is not None:
                capture_timer.cancel()
            stop.set()
            for _, process, frames, statuses in captures:
                process.join(0.15)
                stop_process(process)
                for channel in (frames, statuses):
                    channel.cancel_join_thread()
                    channel.close()

    def _run(self):
        try:
            while not self.stopping.is_set():
                with self.lock:
                    self._prune(make_room=False)
                    jid = self.pending.popleft() if self.pending else None
                    if jid is not None and (
                        jid not in self.jobs or self.jobs[jid]["status"] in TERMINAL
                    ):
                        continue
                    self.active = jid
                if jid is None:
                    self.wake.wait(0.1)
                    self.wake.clear()
                    continue
                try:
                    if not self._interrupted(jid):
                        self._search(jid)
                except Exception as exc:
                    with self.lock:
                        self._finish(jid, "failed", str(exc)[:300])
                    self._reset_detector()
                finally:
                    with self.lock:
                        self.active = None
        finally:
            self._reset_detector()
