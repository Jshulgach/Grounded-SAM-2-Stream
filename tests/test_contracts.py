import json
import math
import subprocess
import sys
from pathlib import Path

import pytest

from grounded_stream.config import Config, DetectorConfig, Source
from grounded_stream.contracts import Detection, Search, targets
from grounded_stream.detector import normalize_detections


@pytest.mark.parametrize(
    "value",
    [None, [], [" "], ["."], ["x" * 121], ["a\n"], [1], ["x"] * 9, [c * 100 for c in "abcde"]],
)
def test_bad_targets(value):
    with pytest.raises(ValueError):
        targets(value)


def test_literal_targets_preserved():
    assert targets([" red backpack. ", "red backpack", "\u73a9\u5177"]) == (
        "red backpack",
        "\u73a9\u5177",
    )


@pytest.mark.parametrize(
    "patch",
    [
        {"duration_seconds": math.nan},
        {"timeout_seconds": math.inf},
        {"duration_seconds": True},
        {"source_ids": ["x", "x"]},
        {"source_ids": ["missing"]},
        {"source_ids": ["../x"]},
        {"duration_seconds": 10, "timeout_seconds": 5},
        {"targets": []},
        {"url": "http://example.com"},
    ],
)
def test_bad_search(patch):
    with pytest.raises(ValueError):
        Search.parse({"targets": ["toy"], "source_ids": ["x"], **patch}, {"x"})


@pytest.mark.parametrize(
    "score,box",
    [
        (math.nan, (0, 0, 1, 1)),
        (0.5, (-0.1, 0, 1, 1)),
        (0.5, (0.8, 0, 0.2, 1)),
        (True, (0, 0, 1, 1)),
    ],
)
def test_bad_detection(score, box):
    with pytest.raises(ValueError):
        Detection("toy", score, box)


def test_normalization_clamps_and_discards_degenerate():
    got = normalize_detections(
        [[-10, 0, 110, 50], [1, 1, 1, 1], [math.nan, 0, 1, 1], [0, 0, 3, 3]],
        [0.7, 0.9, 0.8, math.inf],
        ["toy"] * 4,
        100,
        50,
    )
    assert got == [{"label": "toy", "score": 0.7, "box": (0, 0, 1, 1)}]


@pytest.mark.parametrize(
    "raw",
    [
        {"kind": "webcam", "uri": True},
        {"kind": "webcam", "uri": -1},
        {"kind": "mjpeg", "uri": "file:///tmp/a"},
        {"kind": "mjpeg", "uri": "http://a:b@localhost/x"},
        {"kind": "mjpeg", "uri": "http://localhost/x?token=secret"},
        {"kind": "mjpeg", "uri": "http://localhost:99999/x"},
        {"kind": "webcam", "uri": 0, "sample_fps": math.nan},
        {"kind": "webcam", "uri": 0, "max_width": True},
    ],
)
def test_bad_source(raw, tmp_path):
    with pytest.raises(ValueError):
        Source.parse({"id": "x", **raw}, tmp_path)


def test_config_relative_and_private(tmp_path):
    (tmp_path / "video.mp4").write_bytes(b"fixture")
    path = tmp_path / "config.json"
    path.write_text(json.dumps({"sources": [{"id": "x", "kind": "file", "uri": "video.mp4"}]}))
    config = Config.load(path)
    assert config.sources[0].uri == str(tmp_path / "video.mp4")
    assert "uri" not in config.sources[0].public()


def test_duplicate_sources(tmp_path):
    path = tmp_path / "config.json"
    path.write_text(json.dumps({"sources": [{"id": "x", "kind": "webcam", "uri": 0}] * 2}))
    with pytest.raises(ValueError):
        Config.load(path)


@pytest.mark.parametrize(
    "patch",
    [{"threshold": float("nan")}, {"local_files_only": 1}, {"device": "gpu"}, {"unknown": 1}],
)
def test_bad_detector_config(patch):
    with pytest.raises(ValueError):
        DetectorConfig.parse(patch)


def test_import_has_no_heavy_side_effects():
    code = "import sys; import grounded_stream.cli; assert not {'torch','cv2','transformers','fastapi','ollama','ultralytics','clip'} & set(sys.modules)"
    subprocess.run([sys.executable, "-c", code], check=True, timeout=15)


def test_demo_configuration():
    config = Config.load(Path(__file__).parents[1] / "examples/video.json")
    assert config.sources[0].kind == "file"


@pytest.mark.parametrize("labels", [[], [""]])
def test_empty_detector_result_is_valid(labels):
    assert normalize_detections([], [], labels, 960, 540) == []


def test_nonempty_label_mismatch_still_fails():
    with pytest.raises(ValueError):
        normalize_detections([], [], ["toy"], 960, 540)


def test_huge_number_is_validation_error():
    with pytest.raises(ValueError):
        Search.parse({"targets": ["toy"], "source_ids": ["x"], "duration_seconds": 10**400}, {"x"})
