# CLIP Image Retrieval

Multimodal fashion retrieval service. A fine-tuned CLIP model
(`anhquanlam/clip-finetuned-deepfashion`) is exposed through:

- A **FastAPI** REST API (`/api/v1/...`, Swagger at `/docs`).
- A **Gradio** web UI mounted at `/ui`.
- A **Qdrant** vector database for HNSW cosine similarity search.
- A **MinIO** S3-compatible object store for the image catalogue.

Search by free-text caption ("vintage floral dress with puff sleeves") or by
uploading a reference image — both query types share the same CLIP embedding
space.

## Demo & resources

[![YouTube Project Demo Video](https://img.shields.io/badge/YouTube-Demo_Video-ff0000?logo=youtube)](https://www.youtube.com/watch?v=6h3SuES8a-M)

[![Hugging Face Space](https://img.shields.io/badge/HuggingFace-Space-yellow?logo=huggingface)](https://huggingface.co/spaces/anhquanlam/clip-image-search-app-deepfashion-multimodal)
[![Finetuned Model](https://img.shields.io/badge/HuggingFace-Finetuned_Model-blue?logo=huggingface)](https://huggingface.co/anhquanlam/clip-finetuned-deepfashion)
[![Dataset ZIP](https://img.shields.io/badge/HuggingFace-Dataset-green?logo=huggingface)](https://huggingface.co/datasets/anhquanlam/clip-deepfashion-multimodal/resolve/main/DeepFashion.zip)

## Screenshots

<img width="1919" height="990" alt="image" src="https://github.com/user-attachments/assets/b294222b-1bd7-4b10-bd02-81bee6f878cd" />
<img width="1919" height="989" alt="image" src="https://github.com/user-attachments/assets/39b60ae5-08fb-4797-86f3-23871d39dad7" />
<img width="1917" height="996" alt="image" src="https://github.com/user-attachments/assets/80ebfe3b-0aaf-4a24-919e-e77d90042f36" />

## Repository layout

```
src/
├── api/           # FastAPI factory, dependencies, routes, request/response schemas
├── core/          # EmbeddingService, SearchService, IndexingService, ImageService
├── db/            # Qdrant VectorStore, MinIO ObjectStore, MigrationService
├── ui/            # Gradio Blocks UI factory
├── config.py      # Pydantic Settings (loads from env / .env)
└── server.py      # Entry point — builds app, mounts Gradio, starts uvicorn
docs/
├── architecture.md
├── data-flow.md
└── plans/
scripts/           # Standalone CLIs for one-off migrations
tests/             # pytest suite (Qdrant in-memory, MinIO mock, FastAPI TestClient)
docker-compose.yml # qdrant + minio + app
Dockerfile         # uv-based, multi-layer for fast rebuilds
pyproject.toml     # PEP-621 + uv lockfile (uv.lock)
```

## Prerequisites

- Python ≥ 3.10
- [`uv`](https://docs.astral.sh/uv/) (recommended, ~10× faster than pip).
  Install with `curl -LsSf https://astral.sh/uv/install.sh | sh` or `pipx install uv`.
- Optional, for the full stack: Docker + Docker Compose (Qdrant + MinIO bundled).

## Quick start

### 1) Run the full stack with Docker Compose

```bash
git clone https://github.com/percylam0105/CLIP-Image-Retrieval-Gradio-App.git
cd CLIP-Image-Retrieval-Gradio-App
make docker-up                   # builds + starts qdrant, minio and the app
```

Once `docker compose ps` shows all three services healthy:

| Service | URL | Notes |
|---|---|---|
| Gradio UI | <http://localhost:8000/ui> | Interactive search |
| Swagger / ReDoc | <http://localhost:8000/docs> &nbsp;\|&nbsp; <http://localhost:8000/redoc> | OpenAPI explorer |
| Health probe | <http://localhost:8000/health> | Model + Qdrant + MinIO status |
| MinIO Console | <http://localhost:9001> | Login `minioadmin` / `minioadmin` |
| Qdrant REST | <http://localhost:6333> | `GET /collections` to inspect |

Stop the stack with `make docker-down`. Image / embedding data persists in the
named volumes `qdrant_data` and `minio_data`.

### 2) Run locally (without Docker)

```bash
cp .env.example .env             # optional — defaults to in-memory Qdrant
uv sync                          # creates .venv, installs runtime + dev deps
uv run clip-retrieval            # starts FastAPI on :8000 with Gradio at /ui
```

By default `QDRANT_MODE=memory` so you can start exploring immediately. To
persist data across restarts, change it to `local` (file-based) or `remote`
(point `QDRANT_URL` at a running Qdrant instance). MinIO must be reachable for
image uploads / presigned URLs — easiest is `docker compose up -d minio`.

### 3) Ingest images

Index a directory of images via the REST endpoint:

```bash
curl -X POST http://localhost:8000/api/v1/index/ \
  -H 'Content-Type: application/json' \
  -d '{"images_dir": "/absolute/path/to/images"}'
```

Or use the migration scripts when starting from a pre-computed embedding
dataset:

```bash
uv run python scripts/upload_images_to_minio.py   # uploads images to MinIO
uv run python scripts/migrate_to_qdrant.py        # upserts precomputed embeddings
```

The scripts read `LEGACY_IMAGES_PATH`, `LEGACY_INDEX_PATH`, and `CAPTIONS_PATH`
from your environment / `.env`.

## REST API

| Method | Path | Body | Description |
|---|---|---|---|
| `POST` | `/api/v1/search/text` | `{ "query": str, "top_k": int }` | Text → top-K image hits with presigned URLs |
| `POST` | `/api/v1/search/image` | multipart `file=@...` + `?top_k=N` | Image → top-K similar images |
| `POST` | `/api/v1/index/` | `{ "images_dir": str? }` | Encode + upload + upsert a directory of images |
| `GET` | `/health` | — | Model + Qdrant + MinIO health |

Full schemas live in <docs/architecture.md> and the Swagger UI at `/docs`.

## Configuration

All settings come from environment variables (and an optional `.env`). See
[`.env.example`](.env.example) for the full list. Important ones:

| Variable | Default | Notes |
|---|---|---|
| `MODEL_ID` | `anhquanlam/clip-finetuned-deepfashion` | Hugging Face model id |
| `QDRANT_MODE` | `memory` | `memory` / `local` / `remote` |
| `QDRANT_URL` | `http://localhost:6333` | Used when `QDRANT_MODE=remote` |
| `QDRANT_COLLECTION` | `fashion_images` | Qdrant collection name |
| `MINIO_ENDPOINT` | `localhost:9000` | host:port |
| `MINIO_BUCKET` | `fashion-images` | Bucket; auto-created on startup |
| `API_HOST` / `API_PORT` | `0.0.0.0` / `8000` | uvicorn bind |
| `LOG_LEVEL` | `INFO` | Logging level |

## Development workflow

```bash
make dev          # uv sync — installs runtime + dev deps into .venv
make run          # starts uvicorn + Gradio
make test         # uv run pytest tests/ -v
make lint         # ruff check
make format       # ruff format
make lock         # regenerate uv.lock after editing pyproject.toml
```

The test suite uses Qdrant's in-memory mode and a MinIO mock, so it runs in a
few seconds with no external services.

## Architecture

See [`docs/architecture.md`](docs/architecture.md) for a description of the
service boundaries and module responsibilities, and
[`docs/data-flow.md`](docs/data-flow.md) for Mermaid sequence diagrams of every
request flow (text search, image search, indexing, migration, health).
