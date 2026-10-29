from semantic_cache.cache import SemanticCache
from semantic_cache.eval import run_eval, synthetic_queries


def test_eval_hits_paraphrases_rejects_unrelated():
    report = run_eval(SemanticCache())
    assert report.total == len(synthetic_queries()) == 18
    assert report.false_hits == 0
    assert report.missed_expected == 0
    assert report.hit_rate == 15 / 18
    assert report.cost_saved_usd > 0
    assert report.latency_saved_ms > 0


def test_synthetic_queries_shape():
    queries = synthetic_queries()
    assert sum(1 for _, expected in queries if expected) == 15
    assert sum(1 for _, expected in queries if not expected) == 3
