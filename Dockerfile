FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY file_watcher.py image_classifier.py embedding_cache.py watch_and_classify.py ./


ENV HF_HOME=/app/hf_cache
ENV EMBEDDING_CACHE_PATH=/app/cache/embeddings_cache.db
RUN mkdir -p /app/hf_cache /app/cache /app/watched_folder


ENTRYPOINT ["python", "watch_and_classify.py"]
CMD ["/app/watched_folder", "20"]
