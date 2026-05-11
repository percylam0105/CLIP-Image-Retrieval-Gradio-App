FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# System deps required to build wheels for the CLIP / Qdrant / Pillow stack.
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Python deps first to maximise Docker layer caching.
COPY pyproject.toml README.md ./
COPY src/ src/
RUN pip install --no-cache-dir .

EXPOSE 8000

CMD ["python", "-m", "clip_retrieval"]
