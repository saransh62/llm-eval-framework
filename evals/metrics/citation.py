"""Metric 6 - citation accuracy: cited doc ids must exist and be relevant to the question."""

from __future__ import annotations


def citation_scores(cited: list[str], known_ids: set[str], relevant: list[str]) -> dict[str, float | None]:
    """validity  - share of cited ids that are real corpus documents (None when nothing was cited)
    accuracy  - share of cited ids that are relevant (expected sources); 0 when an answer cites nothing,
                because an uncited answer cannot be verified."""
    if not cited:
        return {"validity": None, "accuracy": 0.0 if relevant else None}
    valid = [c for c in cited if c in known_ids]
    accuracy = sum(1 for c in cited if c in set(relevant)) / len(cited) if relevant else None
    return {"validity": len(valid) / len(cited), "accuracy": accuracy}
