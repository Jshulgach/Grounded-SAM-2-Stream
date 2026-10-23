import pytest
from fastapi.testclient import TestClient

from grounded_stream.runtime import SearchManager
from grounded_stream.service import create_app
from tests.fakes import capture, detector
from tests.test_runtime import config, wait


@pytest.fixture
def client():
    cfg = config("good")
    manager = SearchManager(cfg, capture_target=capture, detector_target=detector)
    with TestClient(create_app(cfg, manager=manager), base_url="http://localhost") as client:
        yield client


def test_idle_service_has_no_models_or_devices(client):
    health = client.get("/v1/health").json()
    assert health["detector"] == "unloaded"
    assert health["active_search"] is None
    assert "uri" not in health["sources"][0]
    assert client.get("/v1/capabilities").json()["segmentation"] is False


def test_packaged_assets_and_schema(client):
    page = client.get("/")
    assert page.status_code == 200 and "Find what matters" in page.text
    assert "frame-ancestors 'none'" in page.headers["content-security-policy"]
    for name in ["app.js", "style.css"]:
        assert client.get("/static/" + name).status_code == 200
    assert client.get("/static/secret").status_code == 404
    schema = client.get("/openapi.json").json()
    assert "requestBody" in schema["paths"]["/v1/searches"]["post"]


@pytest.mark.parametrize(
    "headers",
    [
        {"Origin": "https://evil.example"},
        {"Sec-Fetch-Site": "cross-site"},
        {"Host": "evil.example"},
    ],
)
def test_browser_boundary(client, headers):
    assert client.get("/v1/health", headers=headers).status_code in {400, 403}


def test_body_limits_and_invalid_contract(client):
    assert (
        client.post(
            "/v1/searches", content="{}", headers={"Content-Type": "text/plain"}
        ).status_code
        == 415
    )
    assert (
        client.post(
            "/v1/searches", content="x" * 9000, headers={"Content-Type": "application/json"}
        ).status_code
        == 413
    )
    assert (
        client.post(
            "/v1/searches", json={"targets": ["toy"], "source_ids": ["unknown"]}
        ).status_code
        == 400
    )
    assert (
        client.post(
            "/v1/searches", content="{", headers={"Content-Type": "application/json"}
        ).status_code
        == 400
    )


def test_search_evidence_cancel_and_events(client):
    result = client.post(
        "/v1/searches", json={"targets": ["toy"], "source_ids": ["good"], "duration_seconds": 1}
    )
    assert result.status_code == 202
    jid = result.json()["id"]
    job = wait(client.app.state.manager, jid)
    assert job["status"] == "completed"
    obs = job["observations"][-1]
    assert client.get(obs["evidence_url"]).content.startswith(b"jpeg")
    assert client.get(f"/v1/searches/{jid}/evidence/good?frame_id=old").status_code == 409
    assert client.delete("/v1/searches/" + jid).json()["status"] == "completed"
    assert "event: search" in client.get(f"/v1/searches/{jid}/events").text
    assert client.get("/v1/searches/missing").status_code == 404


def test_token_required_only_for_api():
    cfg = config("good")
    manager = SearchManager(cfg, capture_target=capture, detector_target=detector)
    with TestClient(
        create_app(cfg, manager=manager, token="secret"), base_url="http://localhost"
    ) as client:
        assert client.get("/").status_code == 200
        assert client.get("/v1/health").status_code == 401
        assert (
            client.get("/v1/health", headers={"Authorization": "Bearer secret"}).status_code == 200
        )


def test_continuous_api_capability_and_cancellation(client):
    assert client.get("/v1/capabilities").json()["continuous_search"] is True
    response = client.post(
        "/v1/searches",
        json={
            "targets": ["toy"],
            "source_ids": ["good"],
            "duration_seconds": None,
        },
    )
    assert response.status_code == 202
    job = response.json()
    assert job["request"]["duration_seconds"] is None
    jid = job["id"]
    wait(client.app.state.manager, jid, lambda j: j["processed_frames"] > 0)
    assert client.get("/v1/health").json()["active_search"] == jid
    assert client.delete("/v1/searches/" + jid).json()["status"] == "cancelled"


def test_latest_api_keeps_full_history_available(client):
    job = client.post(
        "/v1/searches", json={"targets": ["toy"], "source_ids": ["good"], "duration_seconds": 1}
    ).json()
    wait(client.app.state.manager, job["id"])
    full = client.get("/v1/searches/" + job["id"]).json()
    latest = client.get("/v1/searches/" + job["id"] + "?latest=true").json()
    assert len(full["observations"]) > 1
    assert latest["observations"] == full["observations"][-1:]
    assert client.get("/v1/capabilities").json()["latest_snapshots"] is True
