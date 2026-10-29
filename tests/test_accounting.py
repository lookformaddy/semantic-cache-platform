from semantic_cache.cache import SemanticCache


def test_hit_rate_and_cost_saved():
    cache = SemanticCache()
    cache.put("How do I reset my password?", "answer", cost_usd=0.02, latency_ms=800.0)
    cache.lookup("How do I reset my password?")  # hit
    cache.lookup("What is the weather on Mars?")  # miss
    stats = cache.accounting.stats("default")
    assert stats.hits == 1 and stats.misses == 1
    assert stats.hit_rate == 0.5
    assert stats.cost_saved_usd == 0.02
    assert stats.latency_saved_ms == 800.0


def test_total_rolls_up_tenants():
    cache = SemanticCache()
    cache.put("How do I reset my password?", "a", namespace="x", cost_usd=0.01)
    cache.lookup("How do I reset my password?", namespace="x")
    cache.lookup("How do I reset my password?", namespace="y")
    total = cache.accounting.total()
    assert total["hits"] == 1 and total["misses"] == 1
