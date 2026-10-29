"""Similarity matching: find the best cached entry for a query."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence

from semantic_cache.embedder import HashingEmbedder
from semantic_cache.store import CacheEntry


@dataclass(frozen=True)
class Match:
    entry: CacheEntry
    similarity: float


def find_best(
    query_embedding: Sequence[float],
    candidates: Sequence[CacheEntry],
    threshold: float,
) -> Optional[Match]:
    """Highest-similarity candidate at or above ``threshold``, else None.

    Ties break on most-recently-accessed, so a fresher duplicate wins.
    """
    best: Optional[Match] = None
    for entry in candidates:
        score = HashingEmbedder.cosine(query_embedding, entry.embedding)
        # 1e-9 tolerance: a true 0.5 cosine can compute as 0.4999999999999999.
        if score < threshold - 1e-9:
            continue
        if best is None or score > best.similarity or (
            score == best.similarity
            and entry.last_accessed_at > best.entry.last_accessed_at
        ):
            best = Match(entry=entry, similarity=score)
    return best


def top_k(
    query_embedding: Sequence[float],
    candidates: Sequence[CacheEntry],
    k: int = 5,
) -> List[Match]:
    """Top-k candidates by similarity, regardless of threshold."""
    scored = [
        Match(entry=e, similarity=HashingEmbedder.cosine(query_embedding, e.embedding))
        for e in candidates
    ]
    scored.sort(key=lambda m: m.similarity, reverse=True)
    return scored[:k]
