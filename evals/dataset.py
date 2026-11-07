"""Golden dataset loading, validation and deterministic stratified sampling."""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from pathlib import Path

DATASET_PATH = Path(__file__).parent / "datasets" / "golden.jsonl"
CATEGORIES = ("factual", "unanswerable", "adversarial")


@dataclass(frozen=True)
class GoldenItem:
    id: str
    category: str
    question: str
    expected_facts: list[str]
    expected_sources: list[str]
    should_refuse: bool
    forbidden: list[str] = field(default_factory=list)
    attack_type: str | None = None
    notes: str = ""

    def __post_init__(self):
        if self.category not in CATEGORIES:
            raise ValueError(f"{self.id}: unknown category {self.category!r}")
        if not self.question.strip():
            raise ValueError(f"{self.id}: empty question")
        if self.category == "factual" and (not self.expected_facts or not self.expected_sources):
            raise ValueError(f"{self.id}: factual items need expected_facts and expected_sources")
        if self.category == "unanswerable" and not self.should_refuse:
            raise ValueError(f"{self.id}: unanswerable items must have should_refuse=true")
        if self.category == "adversarial" and not self.forbidden:
            raise ValueError(f"{self.id}: adversarial items need at least one forbidden string")

    @property
    def answerable(self) -> bool:
        return not self.should_refuse


def load_dataset(path: Path = DATASET_PATH) -> list[GoldenItem]:
    items = [GoldenItem(**json.loads(line)) for line in path.read_text().splitlines() if line.strip()]
    ids = [i.id for i in items]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate ids in golden dataset")
    return items


def sample(items: list[GoldenItem], n: int | None, seed: int = 7) -> list[GoldenItem]:
    """Deterministic sample of n items, stratified by category so small samples still cover every metric."""
    if n is None or n >= len(items):
        return list(items)
    rng = random.Random(seed)
    buckets = {c: [i for i in items if i.category == c] for c in CATEGORIES}
    for bucket in buckets.values():
        rng.shuffle(bucket)
    picked: list[GoldenItem] = []
    while len(picked) < n:
        for c in CATEGORIES:
            if buckets[c] and len(picked) < n:
                picked.append(buckets[c].pop())
    order = {item.id: k for k, item in enumerate(items)}
    return sorted(picked, key=lambda i: order[i.id])
