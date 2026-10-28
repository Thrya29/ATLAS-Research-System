"""Hybrid lexical retrieval over tool descriptions.

BM25 captures exact capability words ("sql", "stock"); a character n-gram
TF-IDF cosine captures morphology and partial matches ("revenues" ~
"revenue"). Rankings are fused with Reciprocal Rank Fusion (Cormack et al.,
2009), which is robust to the two scorers living on different scales.

A neural embedder can be plugged in through ``Embedder``; the default is fully
offline so experiments are reproducible on a laptop.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Sequence
from typing import Protocol

import numpy as np

_TOKEN = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


class BM25:
    def __init__(self, docs: Sequence[str], k1: float = 1.5, b: float = 0.75) -> None:
        self.k1, self.b = k1, b
        self.docs = [tokenize(d) for d in docs]
        self.lengths = np.array([len(d) for d in self.docs], dtype=float)
        self.avgdl = float(self.lengths.mean()) if len(self.docs) else 0.0
        self.tf = [Counter(d) for d in self.docs]
        df: Counter[str] = Counter()
        for d in self.docs:
            df.update(set(d))
        n = len(self.docs)
        self.idf = {w: math.log(1 + (n - f + 0.5) / (f + 0.5)) for w, f in df.items()}

    def scores(self, query: str) -> np.ndarray:
        q = tokenize(query)
        out = np.zeros(len(self.docs))
        for i, tf in enumerate(self.tf):
            norm = self.k1 * (1 - self.b + self.b * self.lengths[i] / (self.avgdl or 1.0))
            s = 0.0
            for w in q:
                f = tf.get(w)
                if f:
                    s += self.idf[w] * f * (self.k1 + 1) / (f + norm)
            out[i] = s
        return out


class Embedder(Protocol):
    def encode(self, texts: Sequence[str]) -> np.ndarray: ...


class CharNgramEmbedder:
    """Hashed character n-gram TF vectors, L2-normalised (no training needed)."""

    def __init__(self, dim: int = 2048, n: tuple[int, ...] = (3, 4)) -> None:
        self.dim, self.n = dim, n

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        m = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, t in enumerate(texts):
            s = f" {' '.join(tokenize(t))} "
            for n in self.n:
                for j in range(len(s) - n + 1):
                    # Stable hash (Python's hash() is salted per process).
                    h = int.from_bytes(s[j : j + n].encode(), "little") % self.dim
                    m[i, h] += 1.0
        norms = np.linalg.norm(m, axis=1, keepdims=True)
        return m / np.maximum(norms, 1e-9)


def rrf(rankings: Sequence[np.ndarray], k: int = 60) -> np.ndarray:
    """Reciprocal rank fusion of score vectors (higher score = better)."""
    fused = np.zeros(len(rankings[0]))
    for scores in rankings:
        order = np.argsort(-scores, kind="stable")
        ranks = np.empty_like(order)
        ranks[order] = np.arange(len(order))
        fused += 1.0 / (k + ranks + 1)
    return fused


class HybridRetriever:
    def __init__(self, names: Sequence[str], docs: Sequence[str], embedder: Embedder | None = None) -> None:
        self.names = list(names)
        self.bm25 = BM25(docs)
        self.embedder = embedder or CharNgramEmbedder()
        self.doc_vecs = self.embedder.encode(docs)

    def scores(self, query: str) -> dict[str, float]:
        if not self.names:
            return {}
        dense = self.doc_vecs @ self.embedder.encode([query])[0]
        fused = rrf([self.bm25.scores(query), dense])
        return dict(zip(self.names, fused.tolist(), strict=True))

    def top_k(self, query: str, k: int) -> list[str]:
        s = self.scores(query)
        return sorted(s, key=lambda n: -s[n])[:k]
