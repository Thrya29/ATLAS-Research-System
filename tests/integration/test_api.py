import pytest
from fastapi.testclient import TestClient


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    import os
    import shutil
    from pathlib import Path

    data = tmp_path_factory.mktemp("data")
    src = Path(__file__).resolve().parents[2] / "data"
    for d in ("docs", "web", "logs"):
        shutil.copytree(src / d, data / d)
    os.environ["ATLAS_DATA_DIR"] = str(data)
    os.environ["ATLAS_API_KEYS"] = "admin-key:admin,analyst-key:analyst"
    from atlas.api.app import app

    with TestClient(app) as c:
        yield c
    assert (data / "state" / "router.json").exists()  # snapshot written on shutdown


ADMIN = {"X-API-Key": "admin-key"}
ANALYST = {"X-API-Key": "analyst-key"}


def test_health(client):
    assert client.get("/healthz").json() == {"status": "ok"}
    assert client.get("/readyz").json()["tools"] == 6


def test_auth_required(client):
    assert client.post("/v1/route", json={"query": "x"}).status_code == 401
    assert client.get("/v1/stats", headers=ANALYST).status_code == 403


def test_route_returns_auditable_decision(client):
    r = client.post("/v1/route", headers=ANALYST, json={"query": "What does our travel policy say about approval steps?"})
    body = r.json()
    assert r.status_code == 200 and r.headers["X-Request-ID"]
    assert [s["tool"] for s in body["plan"]] == ["docs_search"]
    assert {"candidates", "rejected", "estimates", "constraints_satisfied"} <= set(body)


def test_execute_and_feedback(client):
    r = client.post(
        "/v1/execute", headers=ANALYST, json={"query": "Calculate the percentage change in revenue between Q1 and Q2"}
    )
    body = r.json()
    assert body["execution"]["evidence"]["numeric_result"]["output"] is not None
    rid = body["decision"]["request_id"]
    assert client.post("/v1/feedback", headers=ANALYST, json={"request_id": rid, "success": True}).json() == {"accepted": True}
    assert client.post("/v1/feedback", headers=ANALYST, json={"request_id": rid, "success": True}).status_code == 404


def test_caller_supplied_sql_cannot_write(client):
    r = client.post(
        "/v1/execute",
        headers=ADMIN,
        json={"query": "What is today's revenue so far?", "arguments": {"sql_warehouse": {"sql": "DELETE FROM metrics"}}},
    )
    calls = r.json()["execution"]["calls"]
    assert any(c["tool"] == "sql_warehouse" and not c["ok"] for c in calls)


def test_unknown_tool_arguments_rejected(client):
    r = client.post("/v1/execute", headers=ADMIN, json={"query": "x", "arguments": {"rm_rf": {}}})
    assert r.status_code == 422


def test_metrics_exposed(client):
    assert "atlas_decisions_total" in client.get("/metrics").text
