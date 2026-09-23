import os
import sys
import time
import threading
from pathlib import Path


class FileWatcher:
    """
    Watches a folder for image files and tracks changes over time.
    Detects new, modified, and deleted image files.
    """

    IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp', '.tiff', '.svg'}

    def __init__(self, folder, on_new=None, on_modified=None, on_deleted=None, interval=1.0):
        self.folder = Path(folder)
        self.on_new = on_new
        self.on_modified = on_modified
        self.on_deleted = on_deleted
        self.interval = interval

        self.files = {}  # {path: (size, mtime)}
        self.running = False
        self.thread = None

    def _scan(self):
        """Return {path: (size, mtime)} for all images in the folder."""
        found = {}
        try:
            entries = list(self.folder.iterdir())
        except (FileNotFoundError, OSError):
            return found  # folder vanished or inaccessible; treat as empty

        for path in entries:
            try:
                if path.is_file() and path.suffix.lower() in self.IMAGE_EXTENSIONS:
                    stat = path.stat()
                    found[str(path)] = (stat.st_size, stat.st_mtime)
            except (FileNotFoundError, OSError):
                continue  # file disappeared between listing and stat
        return found

    def _check(self):
        current = self._scan()

        for path, meta in current.items():
            if path not in self.files:
                self.files[path] = meta
                if self.on_new:
                    self.on_new(path)
            elif self.files[path] != meta:
                self.files[path] = meta
                if self.on_modified:
                    self.on_modified(path)

        for path in list(self.files):
            if path not in current:
                del self.files[path]
                if self.on_deleted:
                    self.on_deleted(path)

    def _loop(self):
        try:
            self._check()
            while self.running:
                time.sleep(self.interval)
                if self.running:
                    self._check()
        except Exception as e:
            print(f"FileWatcher error: {e}", file=sys.stderr)
            self.running = False

    def start(self):
        """Start watching in the background."""
        self.running = True
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    def stop(self):
        """Stop watching."""
        self.running = False
        if self.thread:
            self.thread.join()


if __name__ == "__main__":
    # Get the path from command-line arguments, not from user input
    if len(sys.argv) < 2:
        print("Usage: python file_watcher.py <folder_path> [watch_duration_seconds]")
        sys.exit(1)

    folder = sys.argv[1]
    duration = int(sys.argv[2]) if len(sys.argv) > 2 else 30

    if not os.path.isdir(folder):
        print(f"Error: '{folder}' is not a valid directory.")
        sys.exit(1)

    watcher = FileWatcher(
        folder,
        on_new=lambda p: print(f"New: {p}"),
        on_modified=lambda p: print(f"Modified: {p}"),
        on_deleted=lambda p: print(f"Deleted: {p}"),
    )

    watcher.start()  # runs in background
    print(f"Watching '{folder}' for {duration} seconds...")

    time.sleep(duration)
    watcher.stop()
