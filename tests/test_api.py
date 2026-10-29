"""FastAPI suite: auto-skipped when FastAPI is not installed, so the
test run passes on a bare system Python as well as in CI."""

import pytest

fastapi = pytest.importorskip("fastapi")

from fastapi.testclient import TestClient  # noqa: E402

from semantic_cache.api import create_app  # noqa: E402
from semantic_cache.cache import SemanticCache  # noqa: E402


def test_health_and_lookup_flow():
    client = TestClient(create_app(SemanticCache()))
    assert client.get("/health").json() == {"status": "ok"}
    client.post("/v1/cache/put", json={
        "prompt": "How do I reset my password?", "response": "Reset it in Settings."})
    hit = client.post("/v1/cache/lookup", json={
        "prompt": "I forgot my password and need to reset it"}).json()
    assert hit["hit"] is True
    miss = client.post("/v1/cache/lookup", json={
        "prompt": "What is the weather on Mars?"}).json()
    assert miss["hit"] is False


def test_warm_and_stats():
    client = TestClient(create_app(SemanticCache()))
    warmed = client.post("/v1/cache/warm", json={
        "pairs": [["How do I reset my password?", "Reset it in Settings."]]}).json()
    assert warmed == {"added": 1}
    assert client.get("/v1/stats").json()["default"]["hits"] == 0
