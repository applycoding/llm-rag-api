.PHONY: install lint test check run compose

install:
	python -m pip install -e ".[dev]"

lint:
	ruff check .
	ruff format --check .

test:
	pytest

check: lint test

run:
	uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

compose:
	docker compose up --build
