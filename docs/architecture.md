# Architecture

## Components

```
                    ┌─────────────────────┐
  prompt ──────────►│ HashingEmbedder     │  content tokens → SHA-256 buckets
                    │ (embedder.py)       │  → L2-normalised vector (dim 256)
                    └────────┬────────────┘
                             │ cosine similarity
                    ┌────────▼────────────┐
                    │ matching.py         │  find_best (threshold) / top_k
                    └────────┬────────────┘
                             │
              ┌──────────────▼──────────────┐
              │ SemanticCache (cache.py)    │  lookup / put / get_or_compute
              │  ├─ CacheStore (store.py)  │  per-namespace dict, TTL, LRU
              │  ├─ Accounting             │  per-tenant hit/miss/savings
              │  └─ StalePolicy (config)   │  serve / refresh / serve+refresh
              └──────────────┬──────────────┘
                             │
              ┌──────────────▼──────────────┐
              │ eval.py / api.py / cli.py   │  harness, FastAPI service, CLI
              └─────────────────────────────┘
```

## Data flow

1. `lookup` embeds the prompt, scans live entries in **one namespace only**,
   and returns the best match at or above the similarity threshold.
2. On a hit, the entry is touched (LRU recency), and accounting records
   the entry's stored cost/latency as saved.
3. On a miss, `get_or_compute` calls the supplied LLM function exactly
   once, stores the result with its cost/latency, and returns it.
4. Every write purges expired entries in that namespace, then evicts
   least-recently-used entries until both caps hold.

## The embedder seam

`HashingEmbedder` is intentionally the weakest link, stated openly:
it is deterministic and dependency-free, which makes the whole system
testable, but it measures lexical overlap, not meaning. Swapping in a
real embedding provider means replacing one class — `SemanticCache`
only calls `embed()` and `cosine()`. The eval harness is the gate for
that swap: re-run it, re-tune the threshold, and check false hits
before shipping.

## Eviction and TTL, precisely

- **Expiry** is hard: `now >= expires_at` means the entry is gone,
  purged lazily on the next read or write in its namespace.
- **Staleness** is soft: past `stale_after_seconds` but before TTL,
  the entry is a hit flagged `stale`, and the configured policy decides
  whether it is served as-is, recomputed before returning, or served
  while a refresh is stored for the next caller.
- **Capacity** is per namespace, enforced on write: entry count, then
  byte size (prompt + response UTF-8 bytes). LRU victim selection uses
  `last_accessed_at`, updated on every hit.

## Failure modes and mitigations

| Failure | Mitigation |
|---|---|
| False hit serves a wrong answer | Conservative threshold + eval harness tracking false hits; threshold is a config, not a constant |
| Tenant data leak | Namespace is part of every store key path; no cross-namespace scan exists |
| Stale answers served forever | TTL is hard; stale policies bound how long a stale hit can live |
| Unbounded memory | Entry and byte caps with LRU eviction on every write |
| Thundering herd on a stale popular entry | `refresh_on_stale` recomputes once per caller in this in-memory design; a distributed lock is the documented next step, not a pretended feature |

## Scaling path (not built, deliberately)

In-memory → Redis for the store + pgvector (or a vector DB) for the
similarity scan, with the same `SemanticCache` interface. The scan in
`matching.py` is O(entries) today, which is correct for the in-memory
scope and the reason a vector index is the first scaling investment.
