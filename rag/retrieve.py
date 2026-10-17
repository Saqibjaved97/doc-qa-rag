"""Three retrievers over the same chunks: TF-IDF, BM25 (implemented here) and a hybrid of both."""
from __future__ import annotations

import math
from collections import Counter

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

from .chunk import Chunk
from .text import tokenize


class TfidfRetriever:
    name = "TF-IDF"

    def __init__(self, chunks: list[Chunk]):
        self.chunks = chunks
        self.vec = TfidfVectorizer(tokenizer=tokenize, lowercase=False, token_pattern=None,
                                   ngram_range=(1, 2), sublinear_tf=True)
        self.matrix = self.vec.fit_transform([c.indexed_text for c in chunks])

    def scores(self, query: str) -> np.ndarray:
        return linear_kernel(self.vec.transform([query]), self.matrix).ravel()

    def search(self, query: str, k: int = 5) -> list[tuple[Chunk, float]]:
        s = self.scores(query)
        top = np.argsort(-s, kind="stable")[:k]
        return [(self.chunks[i], float(s[i])) for i in top]


class BM25Retriever:
    """Okapi BM25 with the usual parameters k1=1.5 and b=0.75."""
    name = "BM25"

    def __init__(self, chunks: list[Chunk], k1: float = 1.5, b: float = 0.75):
        self.chunks, self.k1, self.b = chunks, k1, b
        self.docs = [Counter(tokenize(c.indexed_text)) for c in chunks]
        self.lengths = np.array([sum(d.values()) for d in self.docs], dtype=float)
        self.avg_len = self.lengths.mean()
        df = Counter(t for d in self.docs for t in d)
        n = len(chunks)
        self.idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}

    def scores(self, query: str) -> np.ndarray:
        s = np.zeros(len(self.docs))
        for term in set(tokenize(query)):
            idf = self.idf.get(term)
            if idf is None:
                continue
            tf = np.array([d.get(term, 0) for d in self.docs], dtype=float)
            s += idf * tf * (self.k1 + 1) / (tf + self.k1 * (1 - self.b + self.b * self.lengths / self.avg_len))
        return s

    def search(self, query: str, k: int = 5) -> list[tuple[Chunk, float]]:
        s = self.scores(query)
        top = np.argsort(-s, kind="stable")[:k]
        return [(self.chunks[i], float(s[i])) for i in top]


class HybridRetriever:
    """Reciprocal rank fusion: each retriever votes with 1 / (60 + rank)."""
    name = "Hybrid (TF-IDF + BM25)"

    def __init__(self, chunks: list[Chunk], c: int = 60):
        self.chunks, self.c = chunks, c
        self.parts = [TfidfRetriever(chunks), BM25Retriever(chunks)]

    def scores(self, query: str) -> np.ndarray:
        fused = np.zeros(len(self.chunks))
        for r in self.parts:
            order = np.argsort(-r.scores(query), kind="stable")
            ranks = np.empty(len(order), dtype=int)
            ranks[order] = np.arange(1, len(order) + 1)
            fused += 1.0 / (self.c + ranks)
        return fused

    def search(self, query: str, k: int = 5) -> list[tuple[Chunk, float]]:
        s = self.scores(query)
        top = np.argsort(-s, kind="stable")[:k]
        return [(self.chunks[i], float(s[i])) for i in top]


RETRIEVERS = {"tfidf": TfidfRetriever, "bm25": BM25Retriever, "hybrid": HybridRetriever}
