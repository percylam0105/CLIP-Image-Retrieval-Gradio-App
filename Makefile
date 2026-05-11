.PHONY: install dev run test lint docker-up docker-down migrate

install:
	pip install -e .

dev:
	pip install -e ".[dev]"

run:
	python -m clip_retrieval

test:
	pytest tests/ -v

lint:
	ruff check src/ tests/

docker-up:
	docker compose up -d

docker-down:
	docker compose down

migrate:
	python scripts/migrate_to_qdrant.py
