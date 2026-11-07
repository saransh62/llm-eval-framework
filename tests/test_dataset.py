import re

import pytest

from app.chunking import load_corpus
from app.config import CORPUS_DIR
from evals.dataset import GoldenItem, load_dataset, sample

ITEMS = load_dataset()
CORPUS = load_corpus(CORPUS_DIR)


def _norm(s: str) -> str:
    return re.sub(r"(?<=\d),(?=\d)", "", s.lower())


def test_dataset_shape():
    counts = {c: sum(i.category == c for i in ITEMS) for c in ("factual", "unanswerable", "adversarial")}
    assert len(ITEMS) >= 45
    assert counts["unanswerable"] >= 10 and counts["adversarial"] >= 10


def test_expected_sources_exist():
    for item in ITEMS:
        assert set(item.expected_sources) <= set(CORPUS), item.id


@pytest.mark.parametrize("item", [i for i in ITEMS if i.expected_facts], ids=lambda i: i.id)
def test_every_expected_fact_is_grounded_in_an_expected_source(item):
    """Guards against golden labels that the corpus cannot support (a common silent dataset bug).
    Facts phrased as negations (e.g. 'cannot') are checked against the docs loosely via alternatives."""
    docs = _norm(" ".join(CORPUS[s] for s in item.expected_sources))
    for fact in item.expected_facts:
        assert any(_norm(alt) in docs for alt in fact.split("|")), f"{item.id}: {fact!r} not in sources"


def test_validation_rejects_bad_items():
    with pytest.raises(ValueError):
        GoldenItem("x", "factual", "q?", [], [], False)
    with pytest.raises(ValueError):
        GoldenItem("x", "adversarial", "q?", [], [], True)


def test_sample_is_deterministic_and_stratified():
    s1, s2 = sample(ITEMS, 5), sample(ITEMS, 5)
    assert [i.id for i in s1] == [i.id for i in s2]
    assert {i.category for i in s1} == {"factual", "unanswerable", "adversarial"}
    assert len(sample(ITEMS, None)) == len(ITEMS)
