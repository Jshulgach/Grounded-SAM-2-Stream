import json
from pathlib import Path

import pytest

from grounded_stream.config import Config, DetectorConfig
from grounded_stream.yoloe import YoloE
from tests.test_runtime import submit, wait


@pytest.mark.parametrize(
    "patch",
    [
        {"backend": []},
        {"backend": "unknown"},
        {"image_size": True},
        {"image_size": 0},
        {"image_size": 641},
        {"image_size": float("nan")},
        {"text_encoder": "encoder.ts"},
        {"backend": "yoloe", "model": "model-seg-pf.pt", "text_encoder": "encoder.ts"},
        {"backend": "yoloe", "model": "model.pt"},
        {"backend": "yoloe", "model": "model.pt", "text_encoder": "encoder.ts", "revision": "abc"},
    ],
)
def test_invalid_performance_configuration(patch):
    with pytest.raises(ValueError):
        DetectorConfig.parse(patch)


def test_native_dino_size_remains_default():
    assert DetectorConfig.parse({}).image_size is None
    assert DetectorConfig.parse({}).backend == "grounding_dino"
    assert DetectorConfig.parse({"image_size": 640}).image_size == 640


def test_yoloe_relative_assets_resolve_and_missing_assets_fail(tmp_path):
    body = {
        "sources": [{"id": "desk", "kind": "webcam", "uri": 0}],
        "detector": {"backend": "yoloe", "model": "model.pt", "text_encoder": "encoder.ts"},
    }
    path = tmp_path / "config.json"
    path.write_text(json.dumps(body))
    with pytest.raises(ValueError, match="missing"):
        Config.load(path)
    (tmp_path / "model.pt").write_bytes(b"fixture")
    (tmp_path / "encoder.ts").write_bytes(b"fixture")
    cfg = Config.load(path)
    assert Path(cfg.detector.model) == tmp_path / "model.pt"
    assert Path(cfg.detector.text_encoder) == tmp_path / "encoder.ts"


def test_missing_yoloe_assets_fail_before_model_loading(tmp_path, monkeypatch):
    monkeypatch.setenv("YOLO_AUTOINSTALL", "false")
    monkeypatch.setenv("YOLO_OFFLINE", "true")
    with pytest.raises(ValueError, match="Provision"):
        YoloE(
            DetectorConfig(
                backend="yoloe",
                model=str(tmp_path / "absent.pt"),
                text_encoder=str(tmp_path / "absent.ts"),
            )
        )


def test_latest_snapshot_is_small_complete_per_source_and_independent(managers):
    manager = managers("good", "second")
    jid = submit(manager, duration=None)
    wait(manager, jid, lambda j: j["processed_frames"] >= 50)
    manager.cancel(jid)
    full = manager.snapshot(jid)
    latest = manager.snapshot(jid, latest_only=True)
    assert len(full["observations"]) >= 50
    assert len(latest["observations"]) == 2
    expected = {row["source_id"]: row for row in full["observations"]}
    assert {row["source_id"]: row for row in latest["observations"]} == expected
    assert latest["processed_frames"] == full["processed_frames"]
    assert len(json.dumps(latest)) < len(json.dumps(full)) / 4
    latest["observations"][0]["detections"].clear()
    assert manager.snapshot(jid)["observations"][-1]["detections"]
