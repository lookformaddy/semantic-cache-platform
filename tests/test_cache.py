import pytest

from semantic_cache.cache import SemanticCache
from semantic_cache.config import SemanticCacheConfig, StalePolicy


def _cache(**overrides):
    cfg = SemanticCacheConfig(**overrides) if overrides else SemanticCacheConfig()
    return SemanticCache(config=cfg)


def test_exact_and_paraphrase_hit_unrelated_misses():
    cache = _cache()
    cache.put("How do I reset my password?", "Reset it in Settings.",
              cost_usd=0.02, latency_ms=800.0)
    hit = cache.lookup("I forgot my password and need to reset it")
    assert hit.hit and hit.response == "Reset it in Settings."
    assert cache.lookup("What is the weather on Mars?").hit is False


def test_namespaces_are_isolated():
    cache = _cache()
    cache.put("How do I reset my password?", "Tenant A answer", namespace="a")
    assert cache.lookup("How do I reset my password?", namespace="b").hit is False
    assert cache.lookup("How do I reset my password?", namespace="a").hit


def test_get_or_compute_calls_llm_once():
    cache = _cache()
    calls = []

    def fake_llm(prompt):
        calls.append(prompt)
        return "computed", 0.02, 800.0

    assert cache.get_or_compute("How do I reset my password?", fake_llm).response == "computed"
    assert cache.get_or_compute("I forgot my password and need to reset it", fake_llm).response == "computed"
    assert len(calls) == 1  # second request was a paraphrase hit


def test_ttl_expiry_with_injectable_clock():
    now = [1000.0]
    cache = SemanticCache(
        config=SemanticCacheConfig(ttl_seconds=60, stale_after_seconds=30),
        now=lambda: now[0],
    )
    cache.put("How do I reset my password?", "answer")
    assert cache.lookup("How do I reset my password?").hit
    now[0] = 1061.0
    assert cache.lookup("How do I reset my password?").hit is False


def test_refresh_on_stale_recomputes():
    now = [1000.0]
    cache = SemanticCache(
        config=SemanticCacheConfig(
            ttl_seconds=600, stale_after_seconds=60,
            stale_policy=StalePolicy.REFRESH_ON_STALE),
        now=lambda: now[0],
    )
    cache.put("How do I reset my password?", "old")
    now[0] = 1100.0  # stale, not expired
    result = cache.get_or_compute(
        "How do I reset my password?", lambda p: ("new", 0.01, 100.0))
    assert result.response == "new"
    assert cache.accounting.stats("default").refreshed == 1


def test_serve_until_expire_returns_stale_hit():
    now = [1000.0]
    cache = SemanticCache(
        config=SemanticCacheConfig(
            ttl_seconds=600, stale_after_seconds=60,
            stale_policy=StalePolicy.SERVE_UNTIL_EXPIRE),
        now=lambda: now[0],
    )
    cache.put("How do I reset my password?", "old")
    now[0] = 1100.0
    result = cache.lookup("How do I reset my password?")
    assert result.hit and result.stale and result.response == "old"


def test_warm_skips_duplicates_and_counts():
    cache = _cache()
    pairs = [("How do I reset my password?", "a"), ("How do I reset my password?", "a")]
    assert cache.warm(pairs) == 1
    assert cache.warm(pairs) == 0


def test_invalid_config_rejected():
    with pytest.raises(ValueError):
        SemanticCacheConfig(similarity_threshold=1.5)
    with pytest.raises(ValueError):
        SemanticCacheConfig(stale_after_seconds=99999)
