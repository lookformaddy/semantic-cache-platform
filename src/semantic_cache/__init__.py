"""Semantic caching layer for LLM applications.

Public surface: config, embedder, cache, accounting and eval harness.
No network access is required: embeddings are deterministic hashes.
"""

from semantic_cache.accounting import Accounting, TenantStats
from semantic_cache.cache import CacheResult, SemanticCache
from semantic_cache.config import SemanticCacheConfig, StalePolicy
from semantic_cache.embedder import HashingEmbedder
from semantic_cache.eval import EvalReport, run_eval, synthetic_queries

__all__ = [
    "Accounting",
    "CacheResult",
    "EvalReport",
    "HashingEmbedder",
    "SemanticCache",
    "SemanticCacheConfig",
    "StalePolicy",
    "TenantStats",
    "run_eval",
    "synthetic_queries",
]
__version__ = "0.1.0"
