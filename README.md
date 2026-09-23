# screenshot-agent

Watches a folder for image files and continuously groups them into
categories.
## Project structure

```
screenshot-agent/
├── file_watcher.py         Watches a folder, detects new/modified/deleted images
├── image_classifier.py     CLIP embeddings + HDBSCAN clustering + BLIP auto-labeling
├── embedding_cache.py      SQLite-backed persistence for computed embeddings
├── watch_and_classify.py   Wires everything together; main entry point, runs forever
├── tests/                  pytest suite 
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
├── .dockerignore
├── .gitignore
└── README.md
```

## Persistence

Embeddings are cached to a local SQLite file (`embeddings_cache.db` by
default) keyed by `(path, mtime)`. Restarting the process doesn't
re-embed images it's already seen — only new or modified files hit the
model. Pass `cache_path=None` to `ImageClassifier(...)` to disable this
and keep everything in memory only.

Override the cache location with the `EMBEDDING_CACHE_PATH` environment
variable (this is how the Docker setup points it at a persistent volume).

## Testing

```bash
pip install -r requirements.txt --break-system-packages
pytest tests/ -v
```

## Requirements

- Python 3.9+
- ~1GB free disk for model weights (CLIP + BLIP), downloaded once from
  Hugging Face on first run — needs internet for that first run only
- `torch`, `transformers`, `scikit-learn`, `hdbscan`, `pillow` (see
  `requirements.txt`)

## Running it directly (no Docker)

```bash
pip install -r requirements.txt --break-system-packages
python watch_and_classify.py /path/to/your/photos 20
```

Arguments:

| Arg | Meaning | Default |
|---|---|---|
| `folder` (required) | Directory to watch | — |
| `recluster_interval` | Seconds between re-clustering passes | `15` |

It runs **indefinitely** until you stop it with `Ctrl+C`, at which point it
prints the final classification and exits cleanly.

Example output while running:

```
New image: /photos/beach1.jpg -- embedding...
New image: /photos/dog2.jpg -- embedding...

--- Re-cluster at t=20s ---
[beach] 3 image(s)
[dog] 2 image(s)
[misc] 1 image(s)
```

## Running it standalone (one-off batch, no watching)

To classify a folder once without watching it continuously:

```bash
python image_classifier.py /path/to/folder
```

## Running it with Docker

```bash
mkdir -p watched_folder    # put your images here, or point compose at an existing folder
docker compose up --build
```

This builds the image, mounts `./watched_folder` into the container, and
starts watching it (re-clustering every 30s by default). Model weights
persist in a named volume (`hf_cache`) so restarting or rebuilding the
container doesn't re-download them.

Stop it with:

```bash
docker compose down
```

To point at a different folder or change the re-cluster interval, edit
the `volumes:` and `command:` lines in `docker-compose.yml`, or run
without compose:

```bash
docker build -t screenshot-agent .
docker run -v /path/to/photos:/app/watched_folder \
           -v hf_cache:/app/hf_cache \
           screenshot-agent /app/watched_folder 30
```