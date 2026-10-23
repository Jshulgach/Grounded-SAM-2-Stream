"""CLI usable without model dependencies for help, configuration checks and diagnostics."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import sys
import time
from pathlib import Path

from . import __version__
from .config import Config
from .contracts import TERMINAL, Search
from .runtime import SearchManager


def doctor():
    packages = {}
    for name in [
        "grounded-sam2-stream",
        "torch",
        "torchvision",
        "transformers",
        "opencv-python-headless",
        "Pillow",
        "fastapi",
        "uvicorn",
    ]:
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    return {
        "version": __version__,
        "python": sys.version.split()[0],
        "packages": packages,
        "note": "Metadata only. No devices opened and no model downloaded.",
    }


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Local object search. No camera opens until a search starts."
    )
    parser.add_argument("--version", action="version", version=__version__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("doctor", help="Report installed packages without loading models.")
    validate = sub.add_parser(
        "validate", help="Validate a JSON configuration without opening sources."
    )
    validate.add_argument("--config", required=True)
    run = sub.add_parser(
        "search", help="Search and print final JSON; --continuous runs until interrupted."
    )
    run.add_argument("--config", required=True)
    run.add_argument(
        "--target", action="append", required=True, help="Repeat for several object phrases."
    )
    run.add_argument(
        "--source", action="append", help="Configured source ID; repeat to select several."
    )
    mode = run.add_mutually_exclusive_group()
    mode.add_argument("--duration", type=float, default=10)
    mode.add_argument(
        "--continuous", action="store_true", help="Run until Ctrl+C or all files end."
    )
    run.add_argument("--timeout", type=float, default=180)
    run.add_argument("--output", help="Optional JSON result path (must not already exist).")
    serve = sub.add_parser("serve", help="Run the standalone browser UI on loopback.")
    serve.add_argument("--config", required=True)
    serve.add_argument("--port", type=int, default=8765)
    args = parser.parse_args(argv)
    if args.command == "doctor":
        print(json.dumps(doctor(), indent=2))
        return 0
    try:
        config = Config.load(args.config)
        if args.command == "validate":
            print(
                json.dumps(
                    {"valid": True, "sources": [s.public() for s in config.sources]}, indent=2
                )
            )
            return 0
        if args.command == "serve":
            import uvicorn

            from .service import create_app

            if not 1024 <= args.port <= 65535:
                raise ValueError("Port must be between 1024 and 65535.")
            uvicorn.run(create_app(config), host="127.0.0.1", port=args.port, log_level="warning")
            return 0
        if args.output and Path(args.output).exists():
            raise ValueError("Output already exists; choose a new path.")
        query = Search.parse(
            {
                "targets": args.target,
                "source_ids": args.source or [s.id for s in config.sources],
                "duration_seconds": None if args.continuous else args.duration,
                "timeout_seconds": args.timeout,
            },
            {s.id for s in config.sources},
        )
        runtime = SearchManager(config)
        runtime.start()
        try:
            job = runtime.submit(query)
            while job["status"] not in TERMINAL:
                time.sleep(0.1)
                job = runtime.snapshot(job["id"])
            text = json.dumps(job, indent=2, allow_nan=False)
            if args.output:
                with Path(args.output).open("x", encoding="utf-8") as output:
                    output.write(text + "\n")
            else:
                print(text)
            return 0 if job["status"] == "completed" else 2
        finally:
            runtime.close()
    except KeyboardInterrupt:
        return 130
    except (ValueError, OSError, ImportError, RuntimeError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
