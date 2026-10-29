"""Optional FastAPI service. Importing this module requires FastAPI;
callers that do not have it installed get a clear ImportError and the
core library keeps working. See tests/test_api.py for the skip guard.
"""

from __future__ import annotations

from typing import Optional

try:
    from fastapi import FastAPI
    from pydantic import BaseModel
except ImportError as exc:  # pragma: no cover - exercised via skip test
    raise ImportError("FastAPI is required for the API service: pip install -r requirements.txt") from exc

from semantic_cache.cache import SemanticCache
from semantic_cache.config import SemanticCacheConfig


class LookupRequest(BaseModel):
    prompt: str
    namespace: Optional[str] = None


class PutRequest(BaseModel):
    prompt: str
    response: str
    namespace: Optional[str] = None
    cost_usd: float = 0.0
    latency_ms: float = 0.0


class WarmRequest(BaseModel):
    pairs: list[tuple[str, str]]
    namespace: Optional[str] = None


def create_app(cache: Optional[SemanticCache] = None) -> FastAPI:
    """Build the service around an existing cache (or a fresh one)."""
    app = FastAPI(title="Semantic Cache Platform")
    cache = cache or SemanticCache(SemanticCacheConfig.from_env())

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok"}

    @app.post("/v1/cache/lookup")
    def lookup(req: LookupRequest) -> dict:
        result = cache.lookup(req.prompt, req.namespace)
        return {
            "hit": result.hit,
            "response": result.response,
            "similarity": round(result.similarity, 4),
            "stale": result.stale,
            "cached_prompt": result.cached_prompt,
        }

    @app.post("/v1/cache/put")
    def put(req: PutRequest) -> dict:
        cache.put(req.prompt, req.response, req.namespace, req.cost_usd, req.latency_ms)
        return {"stored": True}

    @app.post("/v1/cache/warm")
    def warm(req: WarmRequest) -> dict:
        added = cache.warm(req.pairs, req.namespace)
        return {"added": added}

    @app.get("/v1/stats")
    def stats(namespace: Optional[str] = None) -> dict:
        return cache.stats(namespace) if namespace else cache.accounting.all_stats()

    return app
