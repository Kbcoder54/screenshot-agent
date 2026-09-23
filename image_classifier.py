from __future__ import annotations

import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

STOPWORDS = {
    "a", "an", "the", "of", "in", "on", "with", "and", "is", "are", "at",
    "to", "there", "this", "that", "it", "its", "some", "several", "few",
    "close", "up", "shot", "photo", "picture", "image", "background",
    "front", "side", "small", "large", "big", "next", "top", "showing",
}


class ImageClassifier:
    def __init__(
        self,
        device: Optional[str] = None,
        min_cluster_size: int = 2,
        num_threads: int = 4,
        max_image_dim: int = 512,
        cache_path: Optional[str] = "embeddings_cache.db",
    ):
        import torch
        from transformers import (
            CLIPModel,
            CLIPProcessor,
            BlipForConditionalGeneration,
            BlipProcessor,
        )

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.min_cluster_size = min_cluster_size
        self.max_image_dim = max_image_dim

        self._torch = torch

        if self.device == "cpu":
            torch.set_num_threads(num_threads)

        self.clip_model = CLIPModel.from_pretrained(
            "openai/clip-vit-base-patch32"
        ).to(self.device).eval()
        self.clip_processor = CLIPProcessor.from_pretrained(
            "openai/clip-vit-base-patch32"
        )

        self.blip_model = BlipForConditionalGeneration.from_pretrained(
            "Salesforce/blip-image-captioning-base"
        ).to(self.device).eval()
        self.blip_processor = BlipProcessor.from_pretrained(
            "Salesforce/blip-image-captioning-base"
        )

        self.cache = None
        if cache_path:
            from embedding_cache import EmbeddingCache

            self.cache = EmbeddingCache(cache_path)

        # path -> embedding vector
        self.embeddings: Dict[str, np.ndarray] = {}
        # path -> cluster id (-1 = unclustered / misc)
        self.assignments: Dict[str, int] = {}
        # cluster id -> human readable label
        self.labels: Dict[int, str] = {}

    # ---------- image loading ----------

    def _load_image(self, path: str) -> Optional[Image.Image]:
        try:
            image = Image.open(path).convert("RGB")
        except Exception:
            return None

        if max(image.size) > self.max_image_dim:
            scale = self.max_image_dim / max(image.size)
            new_size = (round(image.size[0] * scale), round(image.size[1] * scale))
            image = image.resize(new_size, Image.BICUBIC)
        return image

    # ---------- embedding ----------

    def embed(self, path: str) -> Optional[np.ndarray]:
        """
        Compute (or load from cache) a CLIP embedding for one image.
        Returns None on failure (missing/corrupt file).
        """
        try:
            mtime = Path(path).stat().st_mtime
        except OSError:
            mtime = None

        if self.cache is not None and mtime is not None:
            cached = self.cache.get(path, mtime)
            if cached is not None:
                self.embeddings[path] = cached
                return cached

        image = self._load_image(path)
        if image is None:
            return None

        inputs = self.clip_processor(images=image, return_tensors="pt").to(self.device)
        with self._torch.no_grad():
            feats = self.clip_model.get_image_features(**inputs)
        vec = feats.cpu().numpy().flatten()
        vec = vec / (np.linalg.norm(vec) + 1e-8)
        self.embeddings[path] = vec

        if self.cache is not None and mtime is not None:
            self.cache.set(path, mtime, vec)

        return vec

    def embed_many(self, paths: List[str]) -> None:
        for p in paths:
            if p not in self.embeddings:
                self.embed(p)

    def remove(self, path: str) -> None:
        self.embeddings.pop(path, None)
        self.assignments.pop(path, None)
        if self.cache is not None:
            self.cache.delete(path)

    # ---------- clustering ----------

    def recluster(self) -> Dict[str, int]:
        """
        Re-cluster all currently embedded images. Call this after adding/
        removing images (cheap for a few hundred/thousand images; embeddings
        themselves are the expensive part and are cached).
        """
        if len(self.embeddings) < self.min_cluster_size:
            self.assignments = {p: -1 for p in self.embeddings}
            self.labels = {}
            return self.assignments

        try:
            import hdbscan
        except ImportError as e:
            raise ImportError(
                "hdbscan not installed. Run: pip install hdbscan --break-system-packages"
            ) from e

        paths = list(self.embeddings.keys())
        matrix = np.stack([self.embeddings[p] for p in paths])

        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=self.min_cluster_size,
            metric="euclidean",
        )
        cluster_ids = clusterer.fit_predict(matrix)

        self.assignments = {p: int(c) for p, c in zip(paths, cluster_ids)}
        self._label_clusters()
        return self.assignments

    # ---------- labeling ----------

    def _caption(self, path: str) -> str:
        image = self._load_image(path)
        if image is None:
            return ""
        inputs = self.blip_processor(image, return_tensors="pt").to(self.device)
        with self._torch.no_grad():
            out = self.blip_model.generate(**inputs, max_new_tokens=20)
        return self.blip_processor.decode(out[0], skip_special_tokens=True)

    def _label_clusters(self, samples_per_cluster: int = 4) -> None:
        by_cluster: Dict[int, List[str]] = defaultdict(list)
        for path, cid in self.assignments.items():
            by_cluster[cid].append(path)

        self.labels = {}
        for cid, paths in by_cluster.items():
            if cid == -1:
                self.labels[cid] = "misc"
                continue

            sample = paths[:samples_per_cluster]
            captions = [self._caption(p) for p in sample]
            captions = [c for c in captions if c]

            self.labels[cid] = self._summarize_captions(captions) or f"cluster_{cid}"

    @staticmethod
    def _summarize_captions(captions: List[str]) -> str:
        """Pick the most common meaningful word/bigram across captions as the label."""
        if not captions:
            return ""

        words: List[str] = []
        for cap in captions:
            tokens = re.findall(r"[a-zA-Z]+", cap.lower())
            tokens = [t for t in tokens if t not in STOPWORDS and len(t) > 2]
            words.extend(tokens)

        if not words:
            return ""

        counts = Counter(words)
        # Prefer a word that appears in more than one caption if possible.
        common = counts.most_common(3)
        top_word = common[0][0]
        return top_word

    def get_clusters(self) -> Dict[str, Tuple[int, str]]:
        """Return {path: (cluster_id, label)}."""
        return {
            p: (cid, self.labels.get(cid, "unknown"))
            for p, cid in self.assignments.items()
        }

    def summary(self) -> Dict[str, List[str]]:
        """Return {label: [paths]} grouping."""
        out: Dict[str, List[str]] = defaultdict(list)
        for p, cid in self.assignments.items():
            out[self.labels.get(cid, "unknown")].append(p)
        return dict(out)


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python image_classifier.py <folder_path>")
        sys.exit(1)

    folder = Path(sys.argv[1])
    exts = {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp", ".tiff"}
    paths = [str(p) for p in folder.iterdir() if p.suffix.lower() in exts]

    if not paths:
        print(f"No images found in {folder}")
        sys.exit(0)

    print(f"Loading models and embedding {len(paths)} images...")
    clf = ImageClassifier()
    clf.embed_many(paths)

    print("Clustering...")
    clf.recluster()

    print("\nResults:")
    for label, group in clf.summary().items():
        print(f"\n[{label}] ({len(group)} images)")
        for p in group:
            print(f"  {p}")
