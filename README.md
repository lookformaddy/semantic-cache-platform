# Semantic Cache Platform

A semantic caching layer for LLM applications: cache prompt/response pairs
keyed by **embedding similarity**, not exact string match, so a paraphrase
of a previously answered question is served from cache instead of billed
as a fresh LLM call.

## The enterprise problem

LLM calls are slow (hundreds of ms to seconds) and priced per token.
In support, internal-assistant and agent workloads, a large share of
traffic is repeated questions phrased slightly differently. Exact-match
caching misses all of it. Semantic caching closes that gap — but only if
it is safe: a wrong hit (serving an unrelated answer) is worse than a
miss, tenants must never see each other's entries, and stale answers
must be refreshed on a policy, not by accident.

## What it does

- **Similarity matching** — deterministic hashing embedder (no network,
  no weights, no GPU), configurable similarity threshold, top-k lookup.
- **TTL + eviction** — hard expiry, plus LRU eviction under per-namespace
  entry-count and byte caps; expired entries are purged on read/write.
- **Per-tenant namespaces** — fully isolated; a lookup can never hit
  another tenant's entry.
- **Hit/miss/cost accounting** — per-tenant and global hit rate, cost
  saved and latency saved, priced from the cost recorded at first compute.
- **Stale-entry refresh policies** — `serve_until_expire`,
  `refresh_on_stale`, `serve_stale_and_refresh`.
- **Cache warming** — pre-populate from known Q&A; duplicates skipped.
- **Eval harness** — synthetic query set measuring hit rate, false hits,
  missed paraphrases, latency saved and cost saved.
- **Optional FastAPI service** — `/v1/cache/lookup`, `/put`, `/warm`,
  `/stats`, `/health`. The core library works without FastAPI.

## Architecture overview

```
prompt ──► HashingEmbedder ──► cosine match (threshold) ──► CacheStore (per-namespace)
              │                                              │  TTL / LRU / size eviction
              └──► miss ──► LLM compute fn ──► put ─────────┘
hit ──► Accounting (hits, cost saved, latency saved) ──► response
```

See [docs/architecture.md](docs/architecture.md) for the full design,
including the embedder seam, eviction order, and stale-policy semantics.

## How to run it

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt   # core only needs pytest; FastAPI is optional
PYTHONPATH=src python -m semantic_cache.cli demo
PYTHONPATH=src python -m semantic_cache.cli eval
PYTHONPATH=src pytest tests/ -q
```

FastAPI service (requires the optional dependencies):

```bash
pip install -r requirements.txt   # includes fastapi, uvicorn
PYTHONPATH=src python -m semantic_cache.cli serve  # http://localhost:8000
```

Configuration via environment (`SEMANTIC_CACHE_*`, see `.env.example`):
threshold, TTL, stale-after, entry/byte caps, embedding dimension,
default namespace, stale policy.

## Key design decisions and trade-offs

| Decision | Why | Trade-off |
|---|---|---|
| Hashing embedder, no neural model | Zero dependencies, deterministic, testable offline | Captures lexical/paraphrase overlap, not deep semantics |
| Stopword filtering before embedding | Function words ("what is the…") otherwise cause false hits | Drops nuance carried by stopwords in rare cases |
| Threshold 0.5 (not 0.85) | Hashing cosine scales lower than neural cosine; 0.5 is what the eval harness validates | Must be re-tuned if the embedder is swapped — the eval harness exists for exactly that |
| In-memory store | Simple, fast, correct eviction semantics | No persistence or multi-process sharing; the store is the seam for a Redis/pgvector backend |
| Cost saved priced from the original entry | Cannot drift from actual spend; needs no pricing table | Assumes a repeat call would cost the same as the first |
| CI definition shipped as `docs/github-actions-ci.yml` | The GitHub App integration used here cannot write `.github/workflows/` | Activates by moving that file to `.github/workflows/ci.yml` |

## What it demonstrates

Production LLM cost/latency engineering: similarity-based caching with
tenant isolation, TTL/LRU capacity management, stale-refresh policy,
per-tenant savings accounting, and an eval harness that keeps the
threshold honest — the full loop from "cache the LLM" to "prove it saves
money without serving wrong answers".
