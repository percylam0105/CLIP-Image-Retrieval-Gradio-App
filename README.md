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

---

## Beginner's quick start (no Python knowledge required)

> **Audience:** anyone who just wants to *run* the app on their own laptop —
> no Python or AI background needed. Estimated time: 10–15 minutes (plus
> ~5 GB of downloads the first time).
>
> If you already know your way around a terminal, jump to
> [Developer quick start](#developer-quick-start) below.

The simplest way is to use **Docker Desktop**, which runs the whole stack
(the AI model server + the two databases) in containers, so you only need to
install **two** programs on your computer.

### Step 1 — Install the two required programs

Both are free.

1. **Docker Desktop** — runs the application containers.
   - Download: <https://www.docker.com/products/docker-desktop/>
   - Pick the installer for your operating system (Windows / macOS / Linux),
     run it, and follow the prompts. On Windows it will ask to enable WSL 2;
     accept.
   - After installation, **open Docker Desktop once** so it finishes setup,
     and leave it running in the background. You should see a small whale
     icon in your menu bar / system tray when it is ready.

2. **Git** — downloads the project's source code.
   - Download: <https://git-scm.com/downloads>
   - Run the installer with the default options.

> No need to install Python, Node, or anything else — Docker takes care of
> all of that inside the containers.

### Step 2 — Open a terminal

- **Windows:** press the Windows key, type `PowerShell`, press Enter.
- **macOS:** press `Cmd + Space`, type `Terminal`, press Enter.
- **Linux:** open your usual terminal (GNOME Terminal, Konsole, ...).

You will run all the commands below in this window.

### Step 3 — Download the project

Copy/paste the two lines below into the terminal (press Enter after each).

```bash
git clone https://github.com/percylam0105/CLIP-Image-Retrieval-Gradio-App.git
cd CLIP-Image-Retrieval-Gradio-App
```

This creates a folder named `CLIP-Image-Retrieval-Gradio-App` and moves you
inside it.

### Step 4 — Start the application

Run **one** command. It will build the application image the first time you
run it (this takes ~5 minutes and downloads ~3–5 GB — the AI model is large)
and start it in the background.

```bash
docker compose up -d --build
```

While you wait, you should see lines like `=> [internal] load build context`
and `Pulling qdrant`. When the prompt comes back you are done.

To check that everything is running:

```bash
docker compose ps
```

You should see three lines: `app`, `qdrant`, `minio`, all with status `Up`.

> **Tip:** if you have `make` installed (macOS / Linux usually do; on Windows
> install it via [Chocolatey](https://chocolatey.org/) or skip it), you can
> use the shorter `make docker-up` instead of `docker compose up -d --build`.

### Step 5 — Open the app

Open your web browser and visit:

| What you'll see | URL |
|---|---|
| 🖼 The search interface (Gradio) | <http://localhost:8000/ui> |
| 📖 The API documentation (Swagger) | <http://localhost:8000/docs> |
| ❤️ Is everything healthy? | <http://localhost:8000/health> |
| 📂 The image storage admin panel | <http://localhost:8000> *(MinIO console at <http://localhost:9001>, login `minioadmin` / `minioadmin`)* |

The first time you open `/ui`, the AI model is downloaded from Hugging Face —
this takes another minute or two. After that, searches are instant.

> **What if the page doesn't load?** Wait ~30 seconds and refresh — the model
> needs a moment to warm up the first time. If it still doesn't work, see
> [Troubleshooting](#troubleshooting) below.

### Step 6 — Add your own images (optional)

Out of the box the app has no images indexed yet. To add some:

1. Put your `.jpg` / `.jpeg` / `.png` files into a folder on your computer,
   for example `~/my-images`.
2. Tell the app to index them. Replace the path with your actual folder:

   **macOS / Linux:**
   ```bash
   curl -X POST http://localhost:8000/api/v1/index/ \
     -H 'Content-Type: application/json' \
     -d '{"images_dir": "/full/path/to/my-images"}'
   ```

   **Windows PowerShell:**
   ```powershell
   Invoke-RestMethod -Uri http://localhost:8000/api/v1/index/ `
     -Method POST -ContentType 'application/json' `
     -Body '{"images_dir": "C:/full/path/to/my-images"}'
   ```

   (Alternatively, open <http://localhost:8000/docs>, expand
   `POST /api/v1/index/`, click **Try it out**, edit the JSON, and click
   **Execute** — no terminal needed.)

3. Go back to <http://localhost:8000/ui> and search. Your images will now
   appear as results.

### Step 7 — Stop the application

When you're done:

```bash
docker compose down
```

Your indexed images and embeddings are preserved in Docker volumes, so the
next time you run `docker compose up -d` everything is still there.

To **completely wipe** everything (including indexed images), run:

```bash
docker compose down -v
```

### Troubleshooting

| Symptom | Fix |
|---|---|
| `docker: command not found` | Docker Desktop is not installed or not started. Open it from your Applications / Start menu. |
| `Cannot connect to the Docker daemon` | Docker Desktop is installed but not running. Click its icon and wait until it says "Engine running". |
| `port is already allocated` | Another program is using port 8000, 9000, 9001, 6333, or 6334. Either stop the other program or edit `docker-compose.yml` and change the host port (the number on the **left** of `:`). |
| Browser shows "Site can't be reached" | Wait 30 seconds and refresh. The first start has to download the model. Check progress with `docker compose logs -f app`. |
| `out of disk space` / build fails | The model is ~3 GB. Free up ~10 GB of disk space, then re-run the build. |
| Searches return no results | You haven't indexed any images yet. See Step 6, or use the Hugging Face dataset (link at the top). |

---

## Developer quick start

For developers who want to work on the code itself (not just run it).

### Prerequisites

- Python ≥ 3.10
- [`uv`](https://docs.astral.sh/uv/) (recommended, ~10× faster than pip).
  Install with `curl -LsSf https://astral.sh/uv/install.sh | sh` or
  `pipx install uv`.
- Docker + Docker Compose if you want Qdrant + MinIO running locally.

### Run locally without Docker

```bash
git clone https://github.com/percylam0105/CLIP-Image-Retrieval-Gradio-App.git
cd CLIP-Image-Retrieval-Gradio-App
cp .env.example .env             # optional — defaults to in-memory Qdrant
uv sync                          # creates .venv, installs runtime + dev deps
uv run clip-retrieval            # starts FastAPI on :8000 with Gradio at /ui
```

By default `QDRANT_MODE=memory` so you can start exploring immediately. To
persist data across restarts, change it to `local` (file-backed at
`QDRANT_PATH`) or `remote` (point `QDRANT_URL` at a running Qdrant instance).
MinIO must be reachable for image uploads / presigned URLs — easiest is
`docker compose up -d minio`.

### Migrate from a pre-computed dataset

If you already have `df.csv` + `df_image_embeds.npy` + `captions.json`:

```bash
uv run python scripts/upload_images_to_minio.py   # uploads images to MinIO
uv run python scripts/migrate_to_qdrant.py        # upserts precomputed embeddings
```

The scripts read `LEGACY_IMAGES_PATH`, `LEGACY_INDEX_PATH`, and
`CAPTIONS_PATH` from your environment / `.env`.

### Useful make targets

```bash
make dev          # uv sync — installs runtime + dev deps into .venv
make run          # starts uvicorn + Gradio
make test         # uv run pytest tests/ -v
make lint         # ruff check
make format       # ruff format
make lock         # regenerate uv.lock after editing pyproject.toml
make docker-up    # docker compose up -d
make docker-down  # docker compose down
```

The test suite uses Qdrant's in-memory mode and a MinIO mock, so it runs in a
few seconds with no external services.

---

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

## REST API

| Method | Path | Body | Description |
|---|---|---|---|
| `POST` | `/api/v1/search/text` | `{ "query": str, "top_k": int }` | Text → top-K image hits with presigned URLs |
| `POST` | `/api/v1/search/image` | multipart `file=@...` + `?top_k=N` | Image → top-K similar images |
| `POST` | `/api/v1/index/` | `{ "images_dir": str? }` | Encode + upload + upsert a directory of images |
| `GET` | `/health` | — | Model + Qdrant + MinIO health |

Full schemas live in [`docs/architecture.md`](docs/architecture.md) and the
Swagger UI at `/docs`.

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

## Architecture

See [`docs/architecture.md`](docs/architecture.md) for a description of the
service boundaries and module responsibilities, and
[`docs/data-flow.md`](docs/data-flow.md) for Mermaid diagrams of every flow
(startup, text search, image search, indexing, health, migration, UI,
storage shape).
