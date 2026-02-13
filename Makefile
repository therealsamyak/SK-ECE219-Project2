.PHONY: i install test check lint part1 part1-verify

i install:
	uv sync && uv run lefthook install

rebuild:
	rm -rf .venv .pytest_cache .cache __pycache__ && uv sync && uv run lefthook install

test:
	uv run pytest

check lint:
	uv run ruff check --fix . && uv run ruff format .

part1:
	uv run python part1.py

part1-verify:
	@echo "=== Running Part 1 Full Pipeline ==="
	uv run python -c "from part1 import create_length_labels, run_task1_1, run_task1_2, plot_pca_visualizations; print('=== Task 1.1: create_length_labels ==='); df = create_length_labels(); print(f'Reviews: {len(df)}'); print(''); print('=== Task 1.2: run_task1_1 ==='); r1 = run_task1_1(); print('TF-IDF:', r1['tfidf']['matrix_shape']); print('MiniLM:', r1['minilm']['matrix_shape']); print(''); print('=== Task 1.3: run_task1_2 ==='); r2 = run_task1_2(); print('Pipelines run:', len(r2['tfidf']['results']) + len(r2['minilm']['results'])); print(''); print('=== Task 1.4: plot_pca_visualizations ==='); r4 = plot_pca_visualizations(); print('Plot saved:', r4['plot_path'])"
	@echo "=== Part 1 Complete ==="

notebook:
	jupyter lab \
		--port 8888 \
		--IdentityProvider.token "MY_TOKEN" \
		--ip 0.0.0.0