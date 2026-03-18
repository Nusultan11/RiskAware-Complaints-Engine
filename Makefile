.PHONY: install lint test train serve

install:
	pip install -e .[dev]

lint:
	ruff check .
	mypy src

test:
	pytest -q

train:
	python scripts/train.py --config-dir configs

serve:
	python scripts/serve.py

