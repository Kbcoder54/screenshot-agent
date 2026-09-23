"""
Tests for embedding_cache.EmbeddingCache.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))
from embedding_cache import EmbeddingCache


@pytest.fixture
def cache(tmp_path):
    c = EmbeddingCache(str(tmp_path / "cache.db"))
    yield c
    c.close()


def test_set_then_get_returns_same_vector(cache):
    vec = np.array([0.1, 0.2, 0.3], dtype=np.float32)
    cache.set("img.jpg", mtime=100.0, vector=vec)

    result = cache.get("img.jpg", mtime=100.0)
    assert result is not None
    np.testing.assert_allclose(result, vec)


def test_get_missing_path_returns_none(cache):
    assert cache.get("nope.jpg", mtime=100.0) is None


def test_stale_mtime_returns_none(cache):
    vec = np.array([1.0, 2.0], dtype=np.float32)
    cache.set("img.jpg", mtime=100.0, vector=vec)

    # File was modified since caching -- must not serve the stale embedding.
    assert cache.get("img.jpg", mtime=200.0) is None


def test_set_overwrites_existing_entry(cache):
    cache.set("img.jpg", mtime=100.0, vector=np.array([1.0], dtype=np.float32))
    cache.set("img.jpg", mtime=200.0, vector=np.array([2.0], dtype=np.float32))

    result = cache.get("img.jpg", mtime=200.0)
    np.testing.assert_allclose(result, [2.0])
    assert cache.get("img.jpg", mtime=100.0) is None


def test_delete_removes_entry(cache):
    cache.set("img.jpg", mtime=100.0, vector=np.array([1.0], dtype=np.float32))
    cache.delete("img.jpg")
    assert cache.get("img.jpg", mtime=100.0) is None


def test_delete_nonexistent_path_does_not_raise(cache):
    cache.delete("never_existed.jpg")


def test_all_paths(cache):
    cache.set("a.jpg", mtime=1.0, vector=np.array([1.0], dtype=np.float32))
    cache.set("b.jpg", mtime=2.0, vector=np.array([2.0], dtype=np.float32))
    assert sorted(cache.all_paths()) == ["a.jpg", "b.jpg"]


def test_persists_across_reopen(tmp_path):
    db_path = str(tmp_path / "cache.db")
    vec = np.array([1.0, 2.0, 3.0], dtype=np.float32)

    c1 = EmbeddingCache(db_path)
    c1.set("img.jpg", mtime=100.0, vector=vec)
    c1.close()

    c2 = EmbeddingCache(db_path)
    result = c2.get("img.jpg", mtime=100.0)
    c2.close()

    assert result is not None
    np.testing.assert_allclose(result, vec)


def test_context_manager(tmp_path):
    db_path = str(tmp_path / "cache.db")
    with EmbeddingCache(db_path) as c:
        c.set("img.jpg", mtime=1.0, vector=np.array([1.0], dtype=np.float32))
        assert c.get("img.jpg", mtime=1.0) is not None
