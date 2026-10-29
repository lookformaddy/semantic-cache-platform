"""Configuration for the semantic cache, overridable via environment."""

from __future__ import annotations

import os
from dataclasses import dataclass
from enum import Enum


class StalePolicy(str, Enum):
    """What to do when an entry is stale but not yet expired."""

    SERVE_UNTIL_EXPIRE = "serve_until_expire"
    REFRESH_ON_STALE = "refresh_on_stale"
    SERVE_STALE_AND_REFRESH = "serve_stale_and_refresh"


@dataclass(frozen=True)
class SemanticCacheConfig:
    """Tunable cache behaviour. All values are safe defaults for a service.

    similarity_threshold: minimum cosine similarity for a cache hit.
        0.85 is deliberately conservative: paraphrases hit, unrelated
        prompts do not. Lower it only with an eval run to back it up.
    ttl_seconds: hard expiry. stale_after_seconds: soft refresh point.
    max_entries / max_bytes: per-namespace capacity; eviction is LRU.
    """

    # 0.5, not 0.85: a hashing embedder's cosine scale differs from a
    # neural embedder's. 0.5 is the value the eval harness (docs) shows
    # hitting paraphrases while rejecting unrelated prompts; re-tune
    # via the eval harness if the embedder is swapped.
    similarity_threshold: float = 0.5
    ttl_seconds: float = 3600.0
    stale_after_seconds: float = 1800.0
    max_entries: int = 10_000
    max_bytes: int = 50_000_000
    embedding_dim: int = 256
    default_namespace: str = "default"
    stale_policy: StalePolicy = StalePolicy.SERVE_STALE_AND_REFRESH

    def __post_init__(self) -> None:
        if not 0.0 <= self.similarity_threshold <= 1.0:
            raise ValueError("similarity_threshold must be in [0, 1]")
        if self.ttl_seconds <= 0 or self.max_entries <= 0 or self.max_bytes <= 0:
            raise ValueError("ttl_seconds, max_entries and max_bytes must be positive")
        if not 0 <= self.stale_after_seconds <= self.ttl_seconds:
            raise ValueError("stale_after_seconds must be in [0, ttl_seconds]")

    @classmethod
    def from_env(cls, prefix: str = "SEMANTIC_CACHE_") -> "SemanticCacheConfig":
        """Build config from environment variables, falling back to defaults."""

        def _float(name: str, default: float) -> float:
            return float(os.environ.get(prefix + name, default))

        def _int(name: str, default: int) -> int:
            return int(os.environ.get(prefix + name, default))

        policy = os.environ.get(
            prefix + "STALE_POLICY", StalePolicy.SERVE_STALE_AND_REFRESH.value
        )
        return cls(
            similarity_threshold=_float("SIMILARITY_THRESHOLD", 0.5),
            ttl_seconds=_float("TTL_SECONDS", 3600.0),
            stale_after_seconds=_float("STALE_AFTER_SECONDS", 1800.0),
            max_entries=_int("MAX_ENTRIES", 10_000),
            max_bytes=_int("MAX_BYTES", 50_000_000),
            embedding_dim=_int("EMBEDDING_DIM", 256),
            default_namespace=os.environ.get(prefix + "NAMESPACE", "default"),
            stale_policy=StalePolicy(policy),
        )
