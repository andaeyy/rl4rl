.PHONY: install test lint check example

install:
	uv sync --extra dev --extra analysis

test:
	uv run pytest

lint:
	uv run ruff check .

check: lint test

example:
	uv run rl4rl summarize data/examples/synthetic_trajectory.jsonl --external-frontier 36

