"""End-to-end tests of the gateway with a fake Ollama (no GPU needed)."""
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "gateway"))

fastapi = pytest.importorskip("fastapi")
httpx = pytest.importorskip("httpx")
from fastapi.testclient import TestClient  # noqa: E402

from app.auth import hash_key  # noqa: E402
from app.config import Settings  # noqa: E402
from app.main import create_app  # noqa: E402

KEY = "llm_test_key"


def fake_ollama(request: httpx.Request) -> httpx.Response:
    if request.url.path == "/api/tags":
        return httpx.Response(200, json={"models": [{"name": "llama3.1:8b"}]})
    if request.url.path == "/v1/models":
        return httpx.Response(200, json={"data": [{"id": "llama3.1:8b"}, {"id": "other:1b"}]})
    if request.url.path == "/v1/chat/completions":
        body = json.loads(request.content)
        return httpx.Response(200, json={
            "id": "chatcmpl-1", "model": body["model"],
            "choices": [{"index": 0, "message": {"role": "assistant", "content": "Bonjour !"}}],
            "usage": {"prompt_tokens": 5, "completion_tokens": 3},
        })
    return httpx.Response(404)


@pytest.fixture()
def client():
    settings = Settings(upstream_url="http://ollama:11434", api_key_hashes=[hash_key(KEY)],
                        allowed_models=["llama3.1:8b"], rate_limit_per_minute=3)
    app = create_app(settings, transport=httpx.MockTransport(fake_ollama))
    with TestClient(app) as c:
        yield c


def auth(key=KEY):
    return {"Authorization": f"Bearer {key}"}


def chat(model="llama3.1:8b"):
    return {"model": model, "messages": [{"role": "user", "content": "Salut"}]}


def test_health_lists_models(client):
    r = client.get("/health")
    assert r.status_code == 200 and r.json()["models"] == ["llama3.1:8b"]


def test_rejects_missing_or_wrong_key(client):
    assert client.post("/v1/chat/completions", json=chat()).status_code == 401
    assert client.post("/v1/chat/completions", json=chat(), headers=auth("nope")).status_code == 401


def test_chat_completion_is_proxied(client):
    r = client.post("/v1/chat/completions", json=chat(), headers=auth())
    assert r.status_code == 200
    assert r.json()["choices"][0]["message"]["content"] == "Bonjour !"


def test_model_allowlist(client):
    r = client.post("/v1/chat/completions", json=chat("other:1b"), headers=auth())
    assert r.status_code == 403


def test_models_endpoint_is_filtered(client):
    r = client.get("/v1/models", headers=auth())
    assert [m["id"] for m in r.json()["data"]] == ["llama3.1:8b"]


def test_rate_limit_returns_429_with_retry_after(client):
    for _ in range(3):
        client.post("/v1/chat/completions", json=chat(), headers=auth())
    r = client.post("/v1/chat/completions", json=chat(), headers=auth())
    assert r.status_code == 429 and int(r.headers["Retry-After"]) >= 1


def test_metrics_count_requests(client):
    client.post("/v1/chat/completions", json=chat(), headers=auth())
    text = client.get("/metrics").text
    assert 'model="llama3.1:8b",status="200"' in text
