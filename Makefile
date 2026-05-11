.PHONY: install dev run test lint docker-build docker-up docker-down docker-logs migrate

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

docker-build:
	docker compose build

docker-up:
	docker compose up -d

docker-down:
	docker compose down

docker-logs:
	docker compose logs -f app

migrate:
	python scripts/migrate_to_qdrant.py
