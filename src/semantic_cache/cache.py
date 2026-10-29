"""SemanticCache: the orchestrator tying embedder, store, matching,
eviction/TTL and accounting together."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable, Iterable, Optional, Tuple

from semantic_cache.accounting import Accounting
from semantic_cache.config import SemanticCacheConfig, StalePolicy
from semantic_cache.embedder import HashingEmbedder
from semantic_cache.matching import find_best
from semantic_cache.store import CacheEntry, CacheStore

ComputeFn = Callable[[str], Tuple[str, float, float]]  # prompt -> (response, cost, latency_ms)


@dataclass
class CacheResult:
    hit: bool
    response: Optional[str]
    similarity: float = 0.0
    stale: bool = False
    cached_prompt: Optional[str] = None


class SemanticCache:
    """Per-tenant semantic cache.

    ``get_or_compute`` is the main entry point: on a miss it calls the
    supplied LLM function once, caches the result, and returns it; on
    a hit it returns the cached response without calling the LLM.
    """

    def __init__(
        self,
        config: Optional[SemanticCacheConfig] = None,
        embedder: Optional[HashingEmbedder] = None,
        store: Optional[CacheStore] = None,
        now: Optional[Callable[[], float]] = None,
    ) -> None:
        self.config = config or SemanticCacheConfig()
        self.embedder = embedder or HashingEmbedder(self.config.embedding_dim)
        self._now = now or time.time
        self.store = store or CacheStore(
            max_entries=self.config.max_entries,
            max_bytes=self.config.max_bytes,
            ttl_seconds=self.config.ttl_seconds,
            now=self._now,
        )
        self.accounting = Accounting()

    # -- primitives ----------------------------------------------------
    def put(
        self,
        prompt: str,
        response: str,
        namespace: Optional[str] = None,
        cost_usd: float = 0.0,
        latency_ms: float = 0.0,
    ) -> CacheEntry:
        ns = namespace or self.config.default_namespace
        now = self._now()
        entry = CacheEntry(
            prompt=prompt,
            response=response,
            embedding=self.embedder.embed(prompt),
            namespace=ns,
            created_at=now,
            expires_at=now + self.config.ttl_seconds,
            cost_usd=cost_usd,
            latency_ms=latency_ms,
        )
        self.store.put(entry)
        return entry

    def lookup(self, prompt: str, namespace: Optional[str] = None) -> CacheResult:
        ns = namespace or self.config.default_namespace
        now = self._now()
        query = self.embedder.embed(prompt)
        match = find_best(query, self.store.entries(ns), self.config.similarity_threshold)
        if match is None:
            self.accounting.record_miss(ns)
            return CacheResult(hit=False, response=None)
        entry = match.entry
        stale = entry.is_stale(now, self.config.stale_after_seconds)
        self.store.touch(entry)
        self.accounting.record_hit(ns, entry.cost_usd, entry.latency_ms)
        if stale:
            self.accounting.record_stale_served(ns)
        return CacheResult(
            hit=True,
            response=entry.response,
            similarity=match.similarity,
            stale=stale,
            cached_prompt=entry.prompt,
        )

    def get_or_compute(
        self,
        prompt: str,
        compute: ComputeFn,
        namespace: Optional[str] = None,
    ) -> CacheResult:
        """Return a cached response or compute, cache and return a fresh one.

        Stale handling follows ``config.stale_policy``:
        - SERVE_UNTIL_EXPIRE: return the stale hit as-is.
        - REFRESH_ON_STALE: recompute, replace the entry, return fresh.
        - SERVE_STALE_AND_REFRESH: return the stale hit, but refresh the
          entry in the background of this call (synchronously here, so
          the next request is fresh — the seam for async refresh).
        """
        ns = namespace or self.config.default_namespace
        result = self.lookup(prompt, ns)
        if not result.hit:
            response, cost, latency = compute(prompt)
            self.put(prompt, response, ns, cost_usd=cost, latency_ms=latency)
            return CacheResult(hit=False, response=response)
        if result.stale:
            if self.config.stale_policy is StalePolicy.REFRESH_ON_STALE:
                response, cost, latency = compute(prompt)
                self.put(prompt, response, ns, cost_usd=cost, latency_ms=latency)
                self.accounting.record_refreshed(ns)
                return CacheResult(hit=True, response=response, similarity=result.similarity)
            if self.config.stale_policy is StalePolicy.SERVE_STALE_AND_REFRESH:
                # Refresh for the next caller, but do not block this one
                # on a fresh answer beyond the compute already done here.
                response, cost, latency = compute(prompt)
                self.put(prompt, response, ns, cost_usd=cost, latency_ms=latency)
                self.accounting.record_refreshed(ns)
        return result

    # -- warming -------------------------------------------------------
    def warm(
        self,
        pairs: Iterable[Tuple[str, str]],
        namespace: Optional[str] = None,
        cost_usd: float = 0.0,
        latency_ms: float = 0.0,
    ) -> int:
        """Pre-populate the cache. Returns the number of entries added."""
        ns = namespace or self.config.default_namespace
        existing = {e.prompt for e in self.store.entries(ns)}
        added = 0
        for prompt, response in pairs:
            if prompt in existing:
                continue
            self.put(prompt, response, ns, cost_usd=cost_usd, latency_ms=latency_ms)
            existing.add(prompt)
            added += 1
        return added

    def stats(self, namespace: Optional[str] = None) -> dict:
        if namespace:
            return self.accounting.stats(namespace).as_dict()
        return self.accounting.total()
