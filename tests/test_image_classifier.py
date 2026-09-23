"""
Tests for image_classifier.ImageClassifier.

ImageClassifier.__init__ eagerly loads torch + CLIP + BLIP, which is slow
and requires a ~1GB model download -- not something a test suite should
depend on. Instead, these tests construct the object with
`object.__new__` and inject the attributes each method actually needs,
exercising the real clustering/labeling/caching logic against fake but
realistic data.
"""

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))
from image_classifier import ImageClassifier
from embedding_cache import EmbeddingCache


def make_classifier(cache_path=None, min_cluster_size=2):
    """Build an ImageClassifier without running its (heavy) __init__."""
    clf = object.__new__(ImageClassifier)
    clf.device = "cpu"
    clf.min_cluster_size = min_cluster_size
    clf.max_image_dim = 512
    clf.embeddings = {}
    clf.assignments = {}
    clf.labels = {}
    clf.cache = EmbeddingCache(cache_path) if cache_path else None
    return clf


# ---------- _summarize_captions (pure logic, no mocking needed) ----------

def test_summarize_captions_picks_most_common_word():
    captions = [
        "a dog running on the grass",
        "a small dog sitting on a chair",
        "a brown dog looking at the camera",
    ]
    label = ImageClassifier._summarize_captions(captions)
    assert label == "dog"


def test_summarize_captions_empty_list_returns_empty_string():
    assert ImageClassifier._summarize_captions([]) == ""


def test_summarize_captions_filters_stopwords():
    captions = ["a photo of the background", "a picture of the front"]
    # every real word here is a stopword -> nothing meaningful left
    assert ImageClassifier._summarize_captions(captions) == ""


# ---------- recluster() ----------

def test_recluster_groups_similar_embeddings():
    clf = make_classifier(min_cluster_size=2)

    # Two tight groups far apart in embedding space.
    rng = np.random.default_rng(0)
    group_a = {f"a{i}.jpg": np.array([1.0, 0.0]) + rng.normal(0, 0.01, 2) for i in range(3)}
    group_b = {f"b{i}.jpg": np.array([0.0, 1.0]) + rng.normal(0, 0.01, 2) for i in range(3)}
    clf.embeddings = {**group_a, **group_b}

    # Avoid needing BLIP for labeling in this test.
    clf._label_clusters = lambda: setattr(
        clf, "labels", {cid: f"cluster_{cid}" for cid in set(clf.assignments.values())}
    )

    assignments = clf.recluster()

    a_clusters = {assignments[p] for p in group_a}
    b_clusters = {assignments[p] for p in group_b}
    assert len(a_clusters) == 1
    assert len(b_clusters) == 1
    assert a_clusters != b_clusters


def test_recluster_below_min_size_marks_everything_misc():
    clf = make_classifier(min_cluster_size=5)
    clf.embeddings = {"a.jpg": np.array([1.0, 0.0]), "b.jpg": np.array([0.0, 1.0])}

    assignments = clf.recluster()
    assert all(cid == -1 for cid in assignments.values())


def test_recluster_empty_embeddings():
    clf = make_classifier()
    assignments = clf.recluster()
    assert assignments == {}


# ---------- _label_clusters() ----------

def test_label_clusters_assigns_misc_to_unclustered():
    clf = make_classifier()
    clf.assignments = {"a.jpg": -1}
    clf._caption = lambda path: "irrelevant"

    clf._label_clusters()
    assert clf.labels[-1] == "misc"


def test_label_clusters_uses_captions_of_real_cluster():
    clf = make_classifier()
    clf.assignments = {"a.jpg": 0, "b.jpg": 0}
    clf._caption = lambda path: "a photo of a cat sitting on a windowsill"

    clf._label_clusters()
    assert clf.labels[0] == "cat"


def test_label_clusters_falls_back_when_no_meaningful_words():
    clf = make_classifier()
    clf.assignments = {"a.jpg": 0}
    clf._caption = lambda path: ""  # captioning failed

    clf._label_clusters()
    assert clf.labels[0] == "cluster_0"


# ---------- remove() ----------

def test_remove_clears_embedding_and_assignment():
    clf = make_classifier()
    clf.embeddings = {"a.jpg": np.array([1.0])}
    clf.assignments = {"a.jpg": 0}

    clf.remove("a.jpg")
    assert "a.jpg" not in clf.embeddings
    assert "a.jpg" not in clf.assignments


def test_remove_also_evicts_from_cache(tmp_path):
    clf = make_classifier(cache_path=str(tmp_path / "c.db"))
    clf.cache.set("a.jpg", mtime=1.0, vector=np.array([1.0], dtype=np.float32))
    clf.embeddings = {"a.jpg": np.array([1.0])}

    clf.remove("a.jpg")
    assert clf.cache.get("a.jpg", mtime=1.0) is None


def test_remove_missing_path_does_not_raise():
    clf = make_classifier()
    clf.remove("never_added.jpg")  # should not raise


# ---------- summary() / get_clusters() ----------

def test_summary_groups_by_label():
    clf = make_classifier()
    clf.assignments = {"a.jpg": 0, "b.jpg": 0, "c.jpg": -1}
    clf.labels = {0: "dog", -1: "misc"}

    summary = clf.summary()
    assert sorted(summary["dog"]) == ["a.jpg", "b.jpg"]
    assert summary["misc"] == ["c.jpg"]


def test_get_clusters_returns_id_and_label_tuples():
    clf = make_classifier()
    clf.assignments = {"a.jpg": 0}
    clf.labels = {0: "dog"}

    clusters = clf.get_clusters()
    assert clusters["a.jpg"] == (0, "dog")


# ---------- embed() caching behavior ----------

def _fake_embed_pipeline(clf, vector):
    """Wire up minimal fakes so embed() can run without real torch/CLIP."""
    import contextlib

    clf._torch = SimpleNamespace(no_grad=contextlib.nullcontext)
    clf.clip_processor = lambda images, return_tensors: SimpleNamespace(
        to=lambda device: {}
    )

    class FakeTensor:
        def cpu(self):
            return self

        def numpy(self):
            return vector.reshape(1, -1)

    clf.clip_model = SimpleNamespace(get_image_features=lambda **kwargs: FakeTensor())
    clf._load_image = lambda path: object()  # any non-None sentinel


def test_embed_writes_through_to_cache(tmp_path):
    img_path = tmp_path / "a.jpg"
    img_path.write_bytes(b"fake")

    clf = make_classifier(cache_path=str(tmp_path / "cache.db"))
    _fake_embed_pipeline(clf, np.array([3.0, 4.0]))

    vec = clf.embed(str(img_path))
    assert vec is not None
    np.testing.assert_allclose(vec, [0.6, 0.8])  # normalized [3,4] -> [0.6,0.8]

    mtime = img_path.stat().st_mtime
    cached = clf.cache.get(str(img_path), mtime)
    np.testing.assert_allclose(cached, [0.6, 0.8], atol=1e-6)


def test_embed_uses_cache_on_second_call_without_recomputing(tmp_path):
    img_path = tmp_path / "a.jpg"
    img_path.write_bytes(b"fake")

    clf = make_classifier(cache_path=str(tmp_path / "cache.db"))
    _fake_embed_pipeline(clf, np.array([1.0, 0.0]))

    clf.embed(str(img_path))

    calls = {"count": 0}

    def exploding_get_image_features(**kwargs):
        calls["count"] += 1
        raise AssertionError("should not recompute -- cache should have served this")

    clf.clip_model = SimpleNamespace(get_image_features=exploding_get_image_features)

    vec = clf.embed(str(img_path))  # same path, same mtime -> should hit cache
    assert calls["count"] == 0
    np.testing.assert_allclose(vec, [1.0, 0.0])


def test_embed_returns_none_for_unreadable_image(tmp_path):
    clf = make_classifier()
    clf._load_image = lambda path: None  # simulate corrupt/unreadable file

    result = clf.embed(str(tmp_path / "missing.jpg"))
    assert result is None
