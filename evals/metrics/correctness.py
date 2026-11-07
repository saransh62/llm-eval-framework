"""Metric 2 - answer correctness: deterministic key-fact coverage (the judge score lives in evals.judge)."""

from __future__ import annotations

import re

_SPACES = re.compile(r"\s+")
_THOUSANDS = re.compile(r"(?<=\d),(?=\d{3})")
_CITATION = re.compile(r"\[[a-z0-9-]+\]")


def normalize(text: str) -> str:
    """Lowercase, drop citations and thousands separators, unify quotes and whitespace."""
    text = _CITATION.sub(" ", text.lower()).replace("’", "'")
    text = _THOUSANDS.sub("", text)
    return _SPACES.sub(" ", text).strip()


def fact_present(fact: str, answer: str) -> bool:
    """A fact may list alternatives separated by '|'; any one of them counts."""
    norm_answer = normalize(answer)
    return any(normalize(alt) in norm_answer for alt in fact.split("|") if alt.strip())


def missing_facts(facts: list[str], answer: str) -> list[str]:
    return [f for f in facts if not fact_present(f, answer)]


def key_fact_coverage(facts: list[str], answer: str) -> float | None:
    if not facts:
        return None
    return 1 - len(missing_facts(facts, answer)) / len(facts)
