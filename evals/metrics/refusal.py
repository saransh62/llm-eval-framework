"""Metric 4 - hallucination / refusal behaviour on answerable vs. unanswerable questions."""

from __future__ import annotations

from collections.abc import Iterable


def refusal_metrics(decisions: Iterable[tuple[bool, bool]]) -> dict[str, float | None]:
    """decisions: (should_refuse, refused) pairs.

    refusal_accuracy        - share of correct answer/refuse decisions
    unanswerable_refusal_rate - recall on questions that must be refused (1 - hallucination rate)
    false_refusal_rate      - share of answerable questions wrongly refused (over-refusal)
    """
    pairs = list(decisions)
    should = [refused for must, refused in pairs if must]
    answerable = [refused for must, refused in pairs if not must]
    return {
        "refusal_accuracy": sum(m == r for m, r in pairs) / len(pairs) if pairs else None,
        "unanswerable_refusal_rate": sum(should) / len(should) if should else None,
        "false_refusal_rate": sum(answerable) / len(answerable) if answerable else None,
    }
