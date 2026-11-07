"""BM25 retrieval over chunks (no vector database)."""

from __future__ import annotations

import re
from dataclasses import dataclass

from rank_bm25 import BM25Okapi

from app.chunking import Chunk

STOPWORDS = frozenset(
    """a an and are as at be by can do does for from has have how i if in is it its me my of on or
    our so that the their there this to was what when where which who why will with you your
    acme cloud""".split()
)
_TOKEN = re.compile(r"[a-z0-9]+(?:\.[0-9]+)?")


def tokenize(text: str) -> list[str]:
    return [t for t in _TOKEN.findall(text.lower()) if t not in STOPWORDS]


@dataclass(frozen=True)
class ScoredChunk:
    chunk: Chunk
    score: float


class BM25Retriever:
    def __init__(self, chunks: list[Chunk]):
        if not chunks:
            raise ValueError("BM25Retriever needs at least one chunk")
        self.chunks = chunks
        self._bm25 = BM25Okapi([tokenize(c.indexed_text) for c in chunks])

    def search(self, query: str, k: int) -> list[ScoredChunk]:
        """Top-k chunks with a positive score, best first (ties broken by corpus order)."""
        tokens = tokenize(query)
        if not tokens or k <= 0:
            return []
        scores = self._bm25.get_scores(tokens)
        ranked = sorted(range(len(self.chunks)), key=lambda i: (-scores[i], i))
        return [ScoredChunk(self.chunks[i], float(scores[i])) for i in ranked[:k] if scores[i] > 0]


def unique_doc_ids(results: list[ScoredChunk]) -> list[str]:
    """Doc ids in rank order, de-duplicated (first occurrence wins)."""
    seen: dict[str, None] = {}
    for r in results:
        seen.setdefault(r.chunk.doc_id, None)
    return list(seen)
