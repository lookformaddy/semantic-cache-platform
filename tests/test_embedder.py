import math

import pytest

from semantic_cache.embedder import HashingEmbedder, content_tokens, tokenize


def test_deterministic_and_normalised():
    e = HashingEmbedder(dim=64)
    a, b = e.embed("reset my password"), e.embed("reset my password")
    assert a == b
    assert len(a) == 64
    assert math.isclose(sum(v * v for v in a), 1.0, rel_tol=1e-9)


def test_identical_is_one_unrelated_is_low():
    e = HashingEmbedder()
    q = e.embed("reset my password")
    assert e.cosine(q, e.embed("reset my password")) == pytest.approx(1.0)
    assert e.cosine(q, e.embed("weather on Mars")) < 0.3


def test_paraphrase_beats_unrelated():
    e = HashingEmbedder()
    base = e.embed("How do I reset my password?")
    para = e.embed("I forgot my password and need to reset it")
    unrelated = e.embed("What is the weather on Mars?")
    assert e.cosine(base, para) > e.cosine(base, unrelated)
    assert e.cosine(base, para) >= 0.5


def test_empty_text_is_zero_vector():
    e = HashingEmbedder()
    assert e.embed("") == [0.0] * e.dim


def test_stopwords_do_not_drive_similarity():
    assert content_tokens("what is the refund policy") == ["refund", "policy"]
    assert tokenize("What is the refund policy?")  # raw tokenize keeps stops


def test_cosine_rejects_mismatched_lengths():
    e = HashingEmbedder()
    with pytest.raises(ValueError):
        e.cosine([1.0], [1.0, 2.0])
