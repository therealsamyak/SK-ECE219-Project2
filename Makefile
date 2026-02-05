.PHONY: i install test check lint

i install:
	uv sync && uv run lefthook install

test:
	uv run pytest

check lint:
	uv run ruff check --fix . && uv run ruff format .