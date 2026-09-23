import os
import signal
import sys
import time

from file_watcher import FileWatcher  # your existing module
from image_classifier import ImageClassifier

_stop_requested = False


def _handle_stop_signal(signum, frame):
    global _stop_requested
    print(f"\nReceived signal {signum}, shutting down...")
    _stop_requested = True


def main():
    if len(sys.argv) < 2:
        print("Usage: python watch_and_classify.py <folder_path> [recluster_interval]")
        sys.exit(1)

    folder = sys.argv[1]
    recluster_interval = int(sys.argv[2]) if len(sys.argv) > 2 else 15

    if not os.path.isdir(folder):
        print(f"Error: '{folder}' is not a valid directory.")
        sys.exit(1)

    signal.signal(signal.SIGTERM, _handle_stop_signal)
    signal.signal(signal.SIGINT, _handle_stop_signal)

    print("Loading CLIP + BLIP models (first run downloads weights)...")
    cache_path = os.environ.get("EMBEDDING_CACHE_PATH", "embeddings_cache.db")
    clf = ImageClassifier(cache_path=cache_path)

    def handle_new(path):
        print(f"New image: {path} -- embedding...")
        clf.embed(path)

    def handle_modified(path):
        print(f"Modified: {path} -- re-embedding...")
        clf.embed(path)

    def handle_deleted(path):
        print(f"Deleted: {path}")
        clf.remove(path)

    watcher = FileWatcher(
        folder,
        on_new=handle_new,
        on_modified=handle_modified,
        on_deleted=handle_deleted,
    )
    watcher.start()

    print(f"Watching '{folder}' indefinitely, re-clustering every {recluster_interval}s. "
          f"Press Ctrl+C to stop.")

    elapsed = 0
    while not _stop_requested:
        slept = 0
        slice_len = min(1, recluster_interval)
        while slept < recluster_interval and not _stop_requested:
            time.sleep(slice_len)
            slept += slice_len

        if _stop_requested:
            break

        elapsed += recluster_interval
        if clf.embeddings:
            clf.recluster()
            print(f"\n--- Re-cluster at t={elapsed}s ---")
            for label, group in clf.summary().items():
                print(f"[{label}] {len(group)} image(s)")

    watcher.stop()

    print("\n=== Final classification ===")
    for label, group in clf.summary().items():
        print(f"\n[{label}] ({len(group)} images)")
        for p in group:
            print(f"  {p}")


if __name__ == "__main__":
    main()
