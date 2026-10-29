"""In-memory cache store with TTL and LRU/size eviction.

One store holds many namespaces (tenants). Namespaces are fully
isolated: a lookup in one tenant can never hit another tenant's entry,
which is the isolation property multi-tenant LLM services depend on.

Eviction order on every write: (1) drop expired entries, (2) while
over ``max_entries`` or ``max_bytes``, drop the least-recently-used
entry in that namespace. ``now`` is injectable so TTL behaviour is
testable without sleeping.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

TimeFn = Callable[[], float]


@dataclass
class CacheEntry:
    """A single cached prompt/response pair."""

    prompt: str
    response: str
    embedding: List[float]
    namespace: str
    created_at: float
    expires_at: float
    cost_usd: float = 0.0
    latency_ms: float = 0.0
    access_count: int = 0
    last_accessed_at: float = 0.0

    def __post_init__(self) -> None:
        if self.last_accessed_at == 0.0:
            self.last_accessed_at = self.created_at

    @property
    def size_bytes(self) -> int:
        return len(self.prompt.encode("utf-8")) + len(self.response.encode("utf-8"))

    def is_expired(self, now: float) -> bool:
        return now >= self.expires_at

    def is_stale(self, now: float, stale_after_seconds: float) -> bool:
        return (now - self.created_at) >= stale_after_seconds


@dataclass
class EvictionStats:
    expired_evicted: int = 0
    lru_evicted: int = 0


class CacheStore:
    """Namespace-partitioned store. Not thread-safe by design: wrap with
    a lock at the service boundary if serving concurrent requests."""

    def __init__(
        self,
        max_entries: int = 10_000,
        max_bytes: int = 50_000_000,
        ttl_seconds: float = 3600.0,
        now: Optional[TimeFn] = None,
    ) -> None:
        self.max_entries = max_entries
        self.max_bytes = max_bytes
        self.ttl_seconds = ttl_seconds
        self._now: TimeFn = now or time.time
        self._entries: Dict[str, Dict[str, CacheEntry]] = {}
        self.stats = EvictionStats()

    def now(self) -> float:
        return self._now()

    def _namespace(self, namespace: str) -> Dict[str, CacheEntry]:
        return self._entries.setdefault(namespace, {})

    def _namespace_bytes(self, namespace: str) -> int:
        return sum(e.size_bytes for e in self._entries.get(namespace, {}).values())

    def put(self, entry: CacheEntry) -> None:
        """Insert (keyed by exact prompt) and enforce capacity."""
        ns = self._namespace(entry.namespace)
        ns[entry.prompt] = entry
        self._evict_expired(entry.namespace)
        while len(ns) > self.max_entries or self._namespace_bytes(entry.namespace) > self.max_bytes:
            victim = min(ns.values(), key=lambda e: e.last_accessed_at)
            del ns[victim.prompt]
            self.stats.lru_evicted += 1

    def _evict_expired(self, namespace: str) -> int:
        now = self.now()
        ns = self._entries.get(namespace, {})
        expired = [k for k, e in ns.items() if e.is_expired(now)]
        for key in expired:
            del ns[key]
        self.stats.expired_evicted += len(expired)
        return len(expired)

    def entries(self, namespace: str) -> List[CacheEntry]:
        """All live entries in a namespace, in no particular order."""
        self._evict_expired(namespace)
        return list(self._entries.get(namespace, {}).values())

    def touch(self, entry: CacheEntry) -> None:
        entry.access_count += 1
        entry.last_accessed_at = self.now()

    def count(self, namespace: str) -> int:
        return len(self.entries(namespace))

    def clear(self, namespace: Optional[str] = None) -> None:
        if namespace is None:
            self._entries.clear()
        else:
            self._entries.pop(namespace, None)
