.PHONY: i install test check lint part1 part2

i install:
	uv sync && uv run lefthook install

rebuild:
	rm -rf .venv .pytest_cache .cache __pycache__ && uv sync && uv run lefthook install

test:
	uv run pytest

check lint:
	uv run ruff check --fix . && uv run ruff format .

part1:
	uv run part1.py

part2:
	uv run part2.py

notebook:
	jupyter lab \
		--port 8888 \
		--IdentityProvider.token "MY_TOKEN" \
		--ip 0.0.0.0