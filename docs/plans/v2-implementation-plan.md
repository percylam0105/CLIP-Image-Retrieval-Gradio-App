# CLIP Image Retrieval V2 — Implementation Plan

## Mục tiêu
Refactor project thành kiến trúc production-ready MVP:
- Qdrant vector database thay thế file-based storage
- MinIO object storage thay thế local filesystem cho ảnh
- FastAPI REST API + Gradio UI
- Docker + Docker Compose đóng gói toàn bộ

## Kiến trúc

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

## Cấu trúc thư mục mục tiêu

```
clip-image-retrieval/
├── src/
│   └── clip_retrieval/
│       ├── __init__.py
│       ├── __main__.py
│       ├── config.py                 # Pydantic Settings
│       ├── core/
│       │   ├── __init__.py
│       │   ├── embedding.py          # CLIPSearcher (từ clip.py)
│       │   ├── search.py             # SearchService (từ db.py)
│       │   ├── indexing.py           # IndexingService (từ clusterer.py + scan_directory)
│       │   ├── image_service.py      # MinIO image upload/download/URL
│       │   └── schemas.py            # Dataclass SearchResult, ImageMeta, etc.
│       ├── db/
│       │   ├── __init__.py
│       │   ├── vector_store.py       # Qdrant wrapper
│       │   ├── object_store.py       # MinIO wrapper
│       │   └── migration.py          # Migrate .npy + .csv → Qdrant + MinIO
│       ├── api/
│       │   ├── __init__.py
│       │   ├── app.py                # FastAPI app factory
│       │   ├── dependencies.py       # DI container
│       │   ├── schemas.py            # Pydantic request/response
│       │   └── routes/
│       │       ├── __init__.py
│       │       ├── search.py
│       │       ├── index.py
│       │       └── health.py
│       └── ui/
│           ├── __init__.py
│           └── gradio_app.py         # Gradio UI mount vào FastAPI
├── Source-huggingface/               # GIỮ NGUYÊN - không thay đổi
├── scripts/
│   ├── migrate_to_qdrant.py
│   └── upload_images_to_minio.py
├── tests/
│   ├── __init__.py
│   ├── conftest.py
│   ├── test_embedding.py
│   ├── test_search.py
│   ├── test_vector_store.py
│   └── test_api.py
├── docker-compose.yml
├── Dockerfile
├── pyproject.toml
├── .env.example
├── Makefile
├── docs/
│   ├── architecture.md
│   └── plans/
│       └── v2-implementation-plan.md
└── README.md
```

## Phases & Branches

### Phase 1: Project scaffolding (branch: `v2/phase1-scaffolding`)
### Phase 2: Core services (branch: `v2/phase2-core-services`)
### Phase 3: Qdrant integration (branch: `v2/phase3-qdrant`)
### Phase 4: MinIO integration (branch: `v2/phase4-minio`)
### Phase 5: FastAPI + Gradio (branch: `v2/phase5-api`)
### Phase 6: Docker + Docker Compose (branch: `v2/phase6-docker`)
### Phase 7: Migration scripts + Tests (branch: `v2/phase7-migration-tests`)

## Branch flow

```
version-2 (base)
  ├── v2/phase1-scaffolding → merge vào version-2
  ├── v2/phase2-core-services → merge vào version-2
  ├── v2/phase3-qdrant → merge vào version-2
  ├── v2/phase4-minio → merge vào version-2
  ├── v2/phase5-api → merge vào version-2
  ├── v2/phase6-docker → merge vào version-2
  └── v2/phase7-migration-tests → merge vào version-2
```

## Lưu ý quan trọng

- Folder `Source-huggingface/` KHÔNG thay đổi trong bất kỳ phase nào — đây là source đang host trên HuggingFace Spaces.
- Folder `Source/` giữ nguyên làm reference, có thể xóa sau khi v2 hoàn chỉnh.
- File `Notebook for finetuning/` giữ nguyên.
- Mỗi phase phải chạy được (ít nhất import không lỗi) trước khi merge.

## Chi tiết các Phase

### Phase 1: Project Scaffolding
- Tạo cấu trúc thư mục `src/clip_retrieval/...` với tất cả `__init__.py`
- Tạo `pyproject.toml` với deps: torch, transformers, qdrant-client, minio, fastapi, uvicorn, gradio, pydantic-settings, pandas, numpy, pillow, tqdm, python-multipart
- Tạo `src/clip_retrieval/config.py` với Pydantic Settings (model_id, qdrant_*, minio_*, legacy paths, api_*)
- Tạo `.env.example` mới ở root
- Tạo `Makefile` (install, dev, run, test, lint, docker-up, docker-down, migrate)

### Phase 2: Core Services
- `core/schemas.py`: dataclass SearchResult, ImageMeta, CollectionInfo
- `core/embedding.py`: EmbeddingService với lazy loading + `@torch.no_grad()`
- `core/search.py`: SearchService.search_by_text / search_by_image
- `core/indexing.py`: IndexingService.index_directory (encode + upload MinIO + upsert Qdrant)
- `core/image_service.py`: ImageService.get_image_url / get_image_bytes

### Phase 3: Qdrant Integration
- `db/vector_store.py`: VectorStore wrapper Qdrant
  - 3 modes: memory / local / remote
  - upsert_batch dùng uuid5(NAMESPACE_URL, key), HNSW m=32 ef_construct=200, BATCH_SIZE=100
  - search: cosine, hnsw_ef=128
  - get_collection_info / delete_collection

### Phase 4: MinIO Integration
- `db/object_store.py`: ObjectStore wrapper MinIO
  - _ensure_bucket
  - upload_file / upload_bytes
  - get_presigned_url / get_object
  - object_exists / list_objects

### Phase 5: FastAPI + Gradio
- `api/schemas.py`: Pydantic models (TextSearchRequest, SearchResponse, IndexRequest, HealthResponse)
- `api/dependencies.py`: DI container với `@lru_cache`
- `api/routes/search.py`: POST /api/v1/search/text, POST /api/v1/search/image
- `api/routes/index.py`: POST /api/v1/index/
- `api/routes/health.py`: GET /health
- `api/app.py`: FastAPI app factory + CORS
- `ui/gradio_app.py`: Gradio UI refactor (bỏ FAISS flag, bỏ Scan Directory button, dùng MinIO URL)
- `__main__.py`: load settings, mount Gradio vào /ui, uvicorn.run

### Phase 6: Docker
- `Dockerfile`: python:3.11-slim, pip install, copy src, EXPOSE 8000
- `docker-compose.yml`: services qdrant, minio, app + volumes
- `.dockerignore`: .env, DeepFashion/, Source-huggingface/, etc.
- Cập nhật `Makefile` thêm lệnh docker

### Phase 7: Migration + Tests
- `scripts/migrate_to_qdrant.py`: đọc df.csv + .npy → upsert Qdrant
- `scripts/upload_images_to_minio.py`: upload ảnh lên MinIO
- `src/clip_retrieval/db/migration.py`: MigrationService.migrate_all / migrate_embeddings_only
- `tests/`: conftest.py + test_embedding / test_vector_store / test_search / test_api
- Cập nhật `docs/architecture.md` mô tả v2
- Cập nhật `README.md` thêm hướng dẫn chạy v2

## Thứ tự thực hiện

1. `git checkout master && git checkout -b version-2`
2. Tạo file plan này, commit, push branch `version-2`
3. Mỗi phase: tạo branch `v2/phaseN-...` từ `version-2`, implement, commit, push, PR → merge vào `version-2`
4. Tiếp tục theo thứ tự phase 1 → 7
