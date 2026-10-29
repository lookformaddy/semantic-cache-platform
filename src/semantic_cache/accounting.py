"""Hit/miss and savings accounting, per tenant and globally.

A hit saves the cost and latency the LLM call would have incurred.
Those figures come from the cached entry itself (the cost/latency
recorded when it was first computed), so accounting needs no model
pricing table and cannot drift from what was actually spent.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict


@dataclass
class TenantStats:
    hits: int = 0
    misses: int = 0
    stale_served: int = 0
    refreshed: int = 0
    cost_saved_usd: float = 0.0
    latency_saved_ms: float = 0.0

    @property
    def requests(self) -> int:
        return self.hits + self.misses

    @property
    def hit_rate(self) -> float:
        return self.hits / self.requests if self.requests else 0.0

    def as_dict(self) -> dict:
        return {
            "hits": self.hits,
            "misses": self.misses,
            "stale_served": self.stale_served,
            "refreshed": self.refreshed,
            "requests": self.requests,
            "hit_rate": round(self.hit_rate, 4),
            "cost_saved_usd": round(self.cost_saved_usd, 4),
            "latency_saved_ms": round(self.latency_saved_ms, 1),
        }


class Accounting:
    """Per-namespace counters with a global roll-up."""

    def __init__(self) -> None:
        self._tenants: Dict[str, TenantStats] = {}

    def _stats(self, namespace: str) -> TenantStats:
        return self._tenants.setdefault(namespace, TenantStats())

    def record_hit(self, namespace: str, cost_usd: float, latency_ms: float) -> None:
        s = self._stats(namespace)
        s.hits += 1
        s.cost_saved_usd += cost_usd
        s.latency_saved_ms += latency_ms

    def record_miss(self, namespace: str) -> None:
        self._stats(namespace).misses += 1

    def record_stale_served(self, namespace: str) -> None:
        self._stats(namespace).stale_served += 1

    def record_refreshed(self, namespace: str) -> None:
        self._stats(namespace).refreshed += 1

    def stats(self, namespace: str) -> TenantStats:
        return self._stats(namespace)

    def all_stats(self) -> Dict[str, dict]:
        return {ns: s.as_dict() for ns, s in self._tenants.items()}

    def total(self) -> dict:
        agg = TenantStats()
        for s in self._tenants.values():
            agg.hits += s.hits
            agg.misses += s.misses
            agg.stale_served += s.stale_served
            agg.refreshed += s.refreshed
            agg.cost_saved_usd += s.cost_saved_usd
            agg.latency_saved_ms += s.latency_saved_ms
        return agg.as_dict()
