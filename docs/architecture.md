# Architecture — CLIP Image Retrieval V2

## Goals

V2 refactors the original Gradio standalone script into a production-ready MVP:

- Replace file-based storage (`df.csv` + `df_image_embeds.npy` + FAISS `.index`) with a **Qdrant vector database**.
- Replace the local image directory with **MinIO object storage**.
- Expose the retrieval engine over both a **FastAPI REST API** and a **Gradio UI** in a single process.
- Package the entire stack with **Docker + docker-compose** so a single `make docker-up` brings up Qdrant + MinIO + the app.

## High-level diagram

```
┌─────────────────────────────────────────────┐
│              Client (Browser)                │
├─────────────┬───────────────────────────────┤
│  Gradio UI  │     Swagger/ReDoc Docs        │
│  /ui        │     /docs                     │
├─────────────┴───────────────────────────────┤
│              FastAPI Application              │
│  ┌─────────┐ ┌──────────┐ ┌──────────────┐ │
│  │ Search  │ │ Indexing │ │   Health     │ │
│  │ Routes  │ │ Routes   │ │   Routes     │ │
│  └────┬────┘ └────┬─────┘ └──────────────┘ │
│       │           │                          │
│  ┌────┴───────────┴──────────────────────┐  │
│  │         Service Layer                  │  │
│  │  EmbeddingService  SearchService       │  │
│  │  IndexingService   ImageService        │  │
│  └────┬───────────────────────┬──────────┘  │
│       │                       │              │
│  ┌────┴────┐            ┌────┴────┐         │
│  │ Qdrant  │            │  MinIO  │         │
│  │ Vector  │            │ Object  │         │
│  │   DB    │            │ Storage │         │
│  └─────────┘            └─────────┘         │
└─────────────────────────────────────────────┘
```

## Package layout (`src/clip_retrieval/`)

| Path | Responsibility |
|---|---|
| `config.py` | Pydantic `Settings` aggregating CLIP / Qdrant / MinIO / API / legacy paths from env + `.env`. |
| `core/schemas.py` | Internal dataclasses: `SearchResult`, `ImageMeta`, `CollectionInfo`. |
| `core/embedding.py` | `EmbeddingService` — lazy-loaded CLIP model with `@torch.no_grad()` text/image feature extraction. |
| `core/search.py` | `SearchService` — coordinates `EmbeddingService` + `VectorStore`; validates input. |
| `core/indexing.py` | `IndexingService` — scans a directory, encodes images, uploads to MinIO, upserts into Qdrant. |
| `core/image_service.py` | `ImageService` — façade exposing presigned URLs and raw bytes. |
| `db/vector_store.py` | `VectorStore` — Qdrant client wrapper (3 modes: memory / local / remote), HNSW cosine collection. |
| `db/object_store.py` | `ObjectStore` — MinIO client wrapper scoped to a bucket. |
| `db/migration.py` | `MigrationService` — bulk migrate legacy `df.csv` + `df_image_embeds.npy` + `captions.json`. |
| `api/app.py` | `create_app()` FastAPI factory with CORS + 3 routers. |
| `api/dependencies.py` | `lru_cache` DI container; FastAPI `Depends` targets. |
| `api/schemas.py` | Pydantic request / response models. |
| `api/routes/*` | `health.py`, `search.py` (`/api/v1/search/{text,image}`), `index.py` (`/api/v1/index/`). |
| `ui/gradio_app.py` | `build_ui(search_service, image_service)` — Gradio Blocks UI, mounted at `/ui`. |
| `__main__.py` | Entry point — wires services, mounts Gradio, starts uvicorn. |

## Data flow

### Text search

1. Client `POST /api/v1/search/text` with `{ "query": "...", "top_k": N }`.
2. `SearchService.search_by_text` calls `EmbeddingService.get_text_features` → 512-d vector.
3. `VectorStore.search` uses `client.query_points` against the `fashion_images` Qdrant collection with cosine distance.
4. Each hit is enriched with a presigned MinIO URL via `ImageService.get_image_url`.
5. Response body: `SearchResponse(results=[SearchResultItem(image_path, image_url, score, caption, filename), …], total, query)`.

### Image search

Same as text, except the request is `multipart/form-data` with an `UploadFile`; the image is decoded with PIL and fed into `EmbeddingService.get_image_features`.

### Indexing

`POST /api/v1/index/` with optional `images_dir`. `IndexingService.index_directory`:

1. Loads captions from `settings.captions_path` if present.
2. Iterates `*.jpg|.jpeg|.png` in the directory.
3. For each image: encode → `ObjectStore.upload_file` → append to batch.
4. Upserts batch into Qdrant via `VectorStore.upsert_batch` (UUID5 ids, batches of 100).

### Migration (legacy V1 data)

`scripts/migrate_to_qdrant.py` and `scripts/upload_images_to_minio.py` (and the underlying `db/migration.py:MigrationService`):

- `upload_images`: walk `legacy_images_path`, upload missing keys (`images/{filename}`) into MinIO.
- `migrate_embeddings_only`: read `df.csv` + `df_image_embeds.npy` and upsert directly into Qdrant (no re-encoding).
- `migrate_all`: do both, in order.

## Deployment

`docker-compose.yml` brings up three services on a single Compose network:

| Service | Image | Ports |
|---|---|---|
| `qdrant` | `qdrant/qdrant:v1.12.1` | `6333` (REST), `6334` (gRPC) |
| `minio` | `minio/minio:latest` | `9000` (API), `9001` (Console UI) |
| `app` | Built from `Dockerfile` | `8000` (FastAPI + Gradio) |

The `app` service reaches `qdrant` and `minio` over the Compose network with `QDRANT_URL=http://qdrant:6333` and `MINIO_ENDPOINT=minio:9000`. Volumes `qdrant_data` and `minio_data` persist state across restarts.

## Configuration

All settings are derived from environment variables (and optionally a local `.env`, see `.env.example`). See `Settings` in `src/clip_retrieval/config.py` for the full list and defaults.
