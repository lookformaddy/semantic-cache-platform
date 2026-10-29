.PHONY: test demo eval serve

test:
	PYTHONPATH=src pytest tests/ -q

demo:
	PYTHONPATH=src python -m semantic_cache.cli demo

eval:
	PYTHONPATH=src python -m semantic_cache.cli eval

serve:
	PYTHONPATH=src python -m semantic_cache.cli serve
