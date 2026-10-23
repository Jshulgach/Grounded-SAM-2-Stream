"""Run the standalone recorded-video search; no camera or LLM is required."""
from pathlib import Path
from grounded_stream.cli import main

if __name__ == "__main__":
    raise SystemExit(main(["search", "--config", str(Path(__file__).parent / "examples/video.json"),
                          "--target", "deer", "--duration", "10"]))
