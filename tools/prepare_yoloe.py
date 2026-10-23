"""Explicitly download and verify the optional YOLOE assets; never called by the service."""

import argparse
import hashlib
import os
import tempfile
import urllib.request
from pathlib import Path

ASSETS = {
    "yoloe-26s-seg.pt": "48f24206bc8680d60cbbfa296b0140da849669b9515058b72f5a945142df0654",
    "mobileclip2_b.ts": "35d7f213e4d75f38514e4656ad3cb91158bd33e3805d8ac349f23b186f66982f",
}
BASE = "https://github.com/ultralytics/assets/releases/download/v8.4.0/"


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def prepare(directory):
    directory.mkdir(parents=True, exist_ok=True)
    for name, expected in ASSETS.items():
        target = directory / name
        if target.exists():
            if digest(target) != expected:
                raise ValueError(f"{name} exists but has a different hash; it was not overwritten.")
            print(f"Verified {name}")
            continue
        with tempfile.NamedTemporaryFile(dir=directory, suffix=".part", delete=False) as stream:
            pending = Path(stream.name)
            try:
                request = urllib.request.Request(
                    BASE + name, headers={"User-Agent": "Grounded-Stream-Setup"}
                )
                with urllib.request.urlopen(request, timeout=30) as response:
                    total = 0
                    while chunk := response.read(1024 * 1024):
                        total += len(chunk)
                        if total > 512 * 1024 * 1024:
                            raise ValueError("Asset exceeded the download size limit.")
                        stream.write(chunk)
                stream.close()
                if digest(pending) != expected:
                    raise ValueError(f"Checksum mismatch for {name}; asset was not installed.")
                if target.exists():
                    raise FileExistsError(target)
                os.replace(pending, target)
                print(f"Installed and verified {name}")
            finally:
                stream.close()
                pending.unlink(missing_ok=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, default=Path("checkpoints"))
    prepare(parser.parse_args().directory)
