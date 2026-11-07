import pytest

from app.chunking import Chunk
from app.retrieval import BM25Retriever, ScoredChunk, tokenize, unique_doc_ids


def test_tokenize_lowercases_drops_stopwords_keeps_decimals():
    assert tokenize("What is the Pro plan's uptime? 99.9% on Acme Cloud") == [
        "pro", "plan", "s", "uptime", "99.9"
    ]


def test_empty_retriever_rejected():
    with pytest.raises(ValueError):
        BM25Retriever([])


@pytest.mark.parametrize(
    "question, expected_doc",
    [
        ("How much does the Pro plan cost per month?", "pricing"),
        ("What uptime does Enterprise guarantee?", "sla"),
        ("How long do refunds take to reach my card?", "refunds"),
        ("What happens if I exceed the API rate limit?", "api-limits"),
        ("Which regions are available?", "regions"),
        ("Is Acme Cloud HIPAA compliant?", "compliance"),
        ("How long are backups retained?", "data-retention"),
    ],
)
def test_top1_document_for_known_questions(retriever, question, expected_doc):
    results = retriever.search(question, k=4)
    assert results[0].chunk.doc_id == expected_doc


def test_results_sorted_and_limited(retriever):
    results = retriever.search("pricing plan cost", k=3)
    assert len(results) == 3
    assert [r.score for r in results] == sorted((r.score for r in results), reverse=True)


def test_no_overlap_returns_nothing(retriever):
    assert retriever.search("zzzz qqqq", k=4) == []
    assert retriever.search("the and of", k=4) == []
    assert retriever.search("pricing", k=0) == []


def test_unique_doc_ids_preserves_rank_order():
    def sc(doc):
        return ScoredChunk(Chunk(f"{doc}#0", doc, doc, doc, "x"), 1.0)

    assert unique_doc_ids([sc("b"), sc("a"), sc("b"), sc("c")]) == ["b", "a", "c"]
