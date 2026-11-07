import pytest
from fastapi.testclient import TestClient

from app.main import app, get_pipeline


@pytest.fixture()
def client(pipeline):
    app.dependency_overrides[get_pipeline] = lambda: pipeline
    yield TestClient(app)
    app.dependency_overrides.clear()


def test_health(client):
    assert client.get("/health").json() == {"status": "ok"}


def test_ask_contract(client):
    resp = client.post("/ask", json={"question": "How much does the Pro plan cost per month?"})
    assert resp.status_code == 200
    body = resp.json()
    assert set(body) == {"answer", "sources", "citations", "refused", "latency_ms", "tokens", "model"}
    assert isinstance(body["answer"], str) and body["answer"]
    assert body["sources"] and set(body["sources"][0]) == {"doc_id", "chunk_id", "score"}
    assert body["sources"][0]["doc_id"] == "pricing"
    assert body["citations"] == ["pricing"]
    assert body["refused"] is False
    assert body["latency_ms"] >= 0
    assert body["tokens"]["total"] == body["tokens"]["input"] + body["tokens"]["output"]


def test_ask_refusal(client):
    body = client.post("/ask", json={"question": "What is the CEO's home address?"}).json()
    assert body["refused"] is True
    assert body["citations"] == []


@pytest.mark.parametrize("payload", [{}, {"question": ""}, {"question": "x" * 2001}, {"q": "hi"}])
def test_ask_validation(client, payload):
    assert client.post("/ask", json=payload).status_code == 422
