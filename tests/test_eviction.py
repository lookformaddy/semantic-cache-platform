from semantic_cache.cache import SemanticCache
from semantic_cache.config import SemanticCacheConfig
from semantic_cache.store import CacheEntry, CacheStore


def test_lru_evicts_least_recently_used():
    cache = SemanticCache(config=SemanticCacheConfig(max_entries=2))
    cache.put("alpha one", "a1")
    cache.put("beta two", "b2")
    cache.lookup("alpha one")  # alpha is now most recently used
    cache.put("gamma three", "c3")
    assert cache.lookup("beta two").hit is False  # evicted
    assert cache.lookup("alpha one").hit
    assert cache.lookup("gamma three").hit


def test_size_cap_evicts():
    cache = SemanticCache(config=SemanticCacheConfig(max_entries=100, max_bytes=30))
    cache.put("alpha one", "x" * 20)
    cache.put("beta two", "y" * 20)
    assert cache.store.count("default") == 1


def test_expired_entries_evicted_on_read():
    now = [1000.0]
    store = CacheStore(ttl_seconds=10, now=lambda: now[0])
    store.put(CacheEntry(prompt="p", response="r", embedding=[1.0], namespace="n",
                         created_at=1000.0, expires_at=1010.0))
    now[0] = 1020.0
    assert store.entries("n") == []
    assert store.stats.expired_evicted == 1
