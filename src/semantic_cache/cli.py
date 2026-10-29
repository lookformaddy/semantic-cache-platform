"""CLI: demo the cache and run the eval harness.

Usage:
    python -m semantic_cache.cli demo
    python -m semantic_cache.cli eval
    python -m semantic_cache.cli serve   # requires FastAPI + uvicorn
"""

from __future__ import annotations

import argparse
import json
import sys

from semantic_cache.cache import SemanticCache
from semantic_cache.config import SemanticCacheConfig
from semantic_cache.eval import run_eval


def _demo() -> int:
    cache = SemanticCache()
    cache.put("How do I reset my password?", "Use Settings > Security > Reset password.",
              cost_usd=0.02, latency_ms=800.0)
    for q in ["How can I change my password?", "What is the weather on Mars?"]:
        r = cache.lookup(q)
        print(f"{q!r} -> hit={r.hit} similarity={r.similarity:.3f} response={r.response!r}")
    print(json.dumps(cache.stats(), indent=2))
    return 0


def _eval() -> int:
    report = run_eval(SemanticCache())
    print(json.dumps(report.as_dict(), indent=2))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="semantic-cache")
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("demo")
    sub.add_parser("eval")
    sub.add_parser("serve")
    args = parser.parse_args(argv)
    if args.cmd == "demo":
        return _demo()
    if args.cmd == "eval":
        return _eval()
    # serve
    try:
        import uvicorn  # type: ignore
        from semantic_cache.api import create_app
    except ImportError:
        print("serve requires FastAPI and uvicorn: pip install -r requirements.txt", file=sys.stderr)
        return 1
    uvicorn.run(create_app(), host="0.0.0.0", port=8000)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
