.PHONY: setup check test lint data features train eval demo clean

setup:
	uv sync --all-extras
	uv run pre-commit install

check: lint test

lint:
	uv run ruff check src tests scripts
	uv run ruff format --check src tests scripts

test:
	uv run pytest -q

data:
	uv run python scripts/download_datasets.py --help

features:
	uv run python -m aigc_detect.features.extract --help

train:
	uv run python -m aigc_detect.models.train --help || echo "(Day 2: trainer lands here)"

eval:
	uv run python -m aigc_detect.eval.evaluate --help || echo "(Day 1: eval harness lands here)"

demo:
	uv run python app_demo.py --help || echo "(Day 1-2: Gradio demo lands here)"

clean:
	rm -rf .pytest_cache .ruff_cache dist build src/*.egg-info
	find src tests scripts -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
