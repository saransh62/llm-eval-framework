"""Metric 1 - retrieval quality against expected source doc ids. Deterministic; no LLM involved."""

from __future__ import annotations


def recall_at_k(retrieved: list[str], expected: list[str], k: int) -> float | None:
    """Fraction of expected doc ids that appear in the first k retrieved doc ids."""
    if not expected:
        return None
    top = set(retrieved[:k])
    return sum(1 for doc in set(expected) if doc in top) / len(set(expected))


def reciprocal_rank(retrieved: list[str], expected: list[str]) -> float | None:
    """1 / rank of the first relevant doc id (0 when none is retrieved)."""
    if not expected:
        return None
    wanted = set(expected)
    for rank, doc in enumerate(retrieved, start=1):
        if doc in wanted:
            return 1.0 / rank
    return 0.0
