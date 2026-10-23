"""Opt-in real inference benchmark. Only recorded-file configurations are accepted."""

import argparse
import json
import statistics
import time
from dataclasses import replace
from pathlib import Path

from grounded_stream.config import Config
from grounded_stream.contracts import TERMINAL, Search
from grounded_stream.runtime import SearchManager


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="examples/video.json")
    parser.add_argument("--output", required=True)
    parser.add_argument("--duration", type=float, default=5)
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists():
        parser.error("Choose a new output file.")
    cfg = Config.load(args.config)
    if any(source.kind != "file" for source in cfg.sources):
        parser.error("Benchmark configurations must contain recorded files only.")
    source = cfg.sources[0]
    cfg = replace(
        cfg,
        sources=(source, replace(source, id=source.id + "-second", name="Second recorded source")),
    )
    runtime = SearchManager(cfg)
    runtime.start()
    results = []
    try:
        cases = [
            ("positive", ("deer",), (source.id,)),
            ("negative", ("fire truck",), (source.id,)),
            ("two_sources", ("deer",), tuple(s.id for s in cfg.sources)),
        ]
        for name, targets, sources in cases:
            jid = runtime.submit(Search(targets, sources, args.duration, 600))["id"]
            timings = []
            while True:
                start = time.perf_counter()
                runtime.health()
                timings.append((time.perf_counter() - start) * 1000)
                job = runtime.snapshot(jid)
                if job["status"] in TERMINAL:
                    break
                time.sleep(0.05)
            results.append(
                {
                    "case": name,
                    "health_p95_ms": sorted(timings)[int(0.95 * (len(timings) - 1))],
                    "health_median_ms": statistics.median(timings),
                    "result": job,
                }
            )
            print(name, job["status"], job["metrics"], flush=True)
    finally:
        runtime.close()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream:
        json.dump(results, stream, indent=2, allow_nan=False)
    return 0 if all(case["result"]["status"] == "completed" for case in results) else 2


if __name__ == "__main__":
    raise SystemExit(main())
