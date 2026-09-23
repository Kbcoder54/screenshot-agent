from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Optional

import numpy as np


class EmbeddingCache:
    def __init__(self, db_path: str = "embeddings_cache.db"):
        self.db_path = db_path
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS embeddings (
                path TEXT PRIMARY KEY,
                mtime REAL NOT NULL,
                dim INTEGER NOT NULL,
                vector BLOB NOT NULL
            )
            """
        )
        self._conn.commit()

    def get(self, path: str, mtime: float) -> Optional[np.ndarray]:
        """Return the cached embedding for `path` if present and not stale, else None."""
        row = self._conn.execute(
            "SELECT mtime, dim, vector FROM embeddings WHERE path = ?", (path,)
        ).fetchone()
        if row is None:
            return None

        cached_mtime, dim, blob = row
        if abs(cached_mtime - mtime) > 1e-6:
            return None

        return np.frombuffer(blob, dtype=np.float32, count=dim).copy()

    def set(self, path: str, mtime: float, vector: np.ndarray) -> None:
        vector = np.asarray(vector, dtype=np.float32)
        self._conn.execute(
            "INSERT OR REPLACE INTO embeddings (path, mtime, dim, vector) VALUES (?, ?, ?, ?)",
            (path, mtime, vector.shape[0], vector.tobytes()),
        )
        self._conn.commit()

    def delete(self, path: str) -> None:
        self._conn.execute("DELETE FROM embeddings WHERE path = ?", (path,))
        self._conn.commit()

    def all_paths(self) -> list[str]:
        rows = self._conn.execute("SELECT path FROM embeddings").fetchall()
        return [r[0] for r in rows]

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> "EmbeddingCache":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
