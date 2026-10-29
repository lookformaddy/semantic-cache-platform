"""Eval harness: hit rate, latency saved and cost saved on a
synthetic query set. No real user data, no network.

Method: a small set of canonical questions, each with paraphrases.
The harness first warms the cache with the canonical forms, then
issues paraphrases (expected hits) and unrelated questions (expected
misses). A threshold that inflates hit rate by matching unrelated
prompts shows up immediately as false hits.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Sequence, Tuple

from semantic_cache.cache import SemanticCache

# (canonical question, paraphrases, canonical answer)
_CANONICAL: List[Tuple[str, List[str], str]] = [
    (
        "How do I reset my password?",
        ["I forgot my password and need to reset it", "Help me reset my password"],
        "Use Settings > Security > Reset password; a link is emailed to you.",
    ),
    (
        "What is the refund policy?",
        ["Tell me about the refund policy", "Can I get a refund under your policy?"],
        "Refunds are available within 30 days of purchase.",
    ),
    (
        "How do I track my order?",
        ["Where can I track my order?", "I want to track my delivery order"],
        "Tracking is available on the Orders page once the item ships.",
    ),
    (
        "What are your support hours?",
        ["What are the customer support hours?", "Tell me the support hours"],
        "Support is available weekdays 9am-6pm Pacific.",
    ),
    (
        "How do I cancel my subscription?",
        ["I want to cancel my subscription", "Can I cancel my subscription plan?"],
        "Cancel anytime under Settings > Subscription; access runs to period end.",
    ),
]

_UNRELATED = [
    "What is the weather on Mars?",
    "Explain quantum entanglement in one sentence.",
    "How many moons does Jupiter have?",
]


def synthetic_queries() -> List[Tuple[str, bool]]:
    """(query, expected_hit_after_warming) for every eval query."""
    out: List[Tuple[str, bool]] = []
    for canonical, paraphrases, _ in _CANONICAL:
        out.append((canonical, True))
        for p in paraphrases:
            out.append((p, True))
    for q in _UNRELATED:
        out.append((q, False))
    return out


@dataclass
class EvalReport:
    total: int
    hits: int
    misses: int
    expected_hits: int
    false_hits: int
    missed_expected: int
    hit_rate: float
    cost_saved_usd: float
    latency_saved_ms: float

    def as_dict(self) -> dict:
        return {
            "total": self.total,
            "hits": self.hits,
            "misses": self.misses,
            "expected_hits": self.expected_hits,
            "false_hits": self.false_hits,
            "missed_expected": self.missed_expected,
            "hit_rate": round(self.hit_rate, 4),
            "cost_saved_usd": round(self.cost_saved_usd, 4),
            "latency_saved_ms": round(self.latency_saved_ms, 1),
        }


def run_eval(
    cache: SemanticCache,
    queries: Sequence[Tuple[str, bool]] | None = None,
    namespace: str = "eval",
    llm_cost_usd: float = 0.02,
    llm_latency_ms: float = 800.0,
) -> EvalReport:
    """Warm with canonical Q&A, then score the query set.

    ``llm_*`` are the assumed per-call cost/latency of the uncached
    path, used to price what each hit saved.
    """
    queries = list(queries) if queries is not None else synthetic_queries()
    cache.warm(
        [(c, a) for c, _, a in _CANONICAL],
        namespace=namespace,
        cost_usd=llm_cost_usd,
        latency_ms=llm_latency_ms,
    )

    def _fake_llm(prompt: str) -> Tuple[str, float, float]:
        return f"Answer to: {prompt}", llm_cost_usd, llm_latency_ms

    hits = misses = false_hits = missed_expected = 0
    for query, expected in queries:
        # Use lookup (not get_or_compute) so a miss is not cached and
        # cannot inflate later queries in the same run.
        result = cache.lookup(query, namespace)
        if result.hit:
            hits += 1
            if not expected:
                false_hits += 1
        else:
            misses += 1
            if expected:
                missed_expected += 1
    stats = cache.accounting.stats(namespace)
    return EvalReport(
        total=len(queries),
        hits=hits,
        misses=misses,
        expected_hits=sum(1 for _, e in queries if e),
        false_hits=false_hits,
        missed_expected=missed_expected,
        hit_rate=hits / len(queries) if queries else 0.0,
        cost_saved_usd=stats.cost_saved_usd,
        latency_saved_ms=stats.latency_saved_ms,
    )
