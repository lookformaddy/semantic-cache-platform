"""Deterministic, offline embedding for semantic matching.

This is deliberately NOT a neural embedding model: it runs with no
network, no weights and no GPU, so the cache is testable and
reproducible anywhere. Tokens are hashed (SHA-256) into a fixed-size
vector, counts are accumulated, and the vector is L2-normalised so
cosine similarity is a plain dot product.

Trade-off, stated plainly: hashing embeddings capture lexical overlap
and paraphrase structure well, but not deep semantics. In production
this module is the seam where a real embedding provider plugs in —
swap ``HashingEmbedder`` for an API-backed embedder and nothing else
changes. The eval harness exists to quantify that swap.
"""

from __future__ import annotations

import hashlib
import math
import re
from typing import Sequence

_TOKEN_RE = re.compile(r"[a-z0-9]+")

# Function words carry no topic signal and would let unrelated prompts
# match on "what is the ...". Content tokens drive similarity.
STOPWORDS = frozenset(
    "a an the is are was were do does did can could should would "
    "how what when where why which who whom i my me you your he she "
    "it its we our they their of for to in on at by with from please "
    "tell me want need help get".split()
)


def tokenize(text: str) -> list[str]:
    """Lowercase alphanumeric tokens; empty input yields no tokens."""
    return _TOKEN_RE.findall(text.lower())


def content_tokens(text: str) -> list[str]:
    """Tokens minus stopwords; falls back to raw tokens if all are stops."""
    tokens = [t for t in tokenize(text) if t not in STOPWORDS]
    return tokens or tokenize(text)


class HashingEmbedder:
    """Deterministic bag-of-hashed-tokens embedder."""

    def __init__(self, dim: int = 256) -> None:
        if dim <= 0:
            raise ValueError("dim must be positive")
        self.dim = dim

    def embed(self, text: str) -> list[float]:
        """Return an L2-normalised vector of length ``dim``."""
        vec = [0.0] * self.dim
        for token in content_tokens(text):
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            bucket = int.from_bytes(digest[:4], "big") % self.dim
            # Unsigned counts on purpose: signed hashing can cancel to
            # a zero vector when two tokens collide in one bucket with
            # opposite signs. Collisions here only add weight, never erase.
            vec[bucket] += 1.0
        norm = math.sqrt(sum(v * v for v in vec))
        if norm == 0.0:
            return vec
        return [v / norm for v in vec]

    @staticmethod
    def cosine(a: Sequence[float], b: Sequence[float]) -> float:
        """Cosine similarity; inputs from :meth:`embed` are already unit."""
        if len(a) != len(b):
            raise ValueError("vectors must have equal length")
        return sum(x * y for x, y in zip(a, b))
