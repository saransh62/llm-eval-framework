from app.chunking import chunk_corpus, chunk_markdown, load_corpus
from app.config import CORPUS_DIR

DOC = """# Title

Intro sentence.

## Section A
First fact. Second fact.

## Empty

## Section B
""" + " ".join(f"Sentence number {i} is here." for i in range(40))


def test_sections_become_chunks_with_ids():
    chunks = chunk_markdown("doc", DOC, max_chars=10_000)
    assert [c.section for c in chunks] == ["Title", "Section A", "Section B"]
    assert [c.chunk_id for c in chunks] == ["doc#0", "doc#1", "doc#2"]
    assert all(c.doc_id == "doc" and c.title == "Title" for c in chunks)


def test_empty_sections_are_skipped():
    assert "Empty" not in {c.section for c in chunk_markdown("doc", DOC)}


def test_long_sections_split_on_sentence_boundaries():
    chunks = [c for c in chunk_markdown("doc", DOC, max_chars=120) if c.section == "Section B"]
    assert len(chunks) > 1
    assert all(len(c.text) <= 120 for c in chunks)
    assert all(c.text.endswith(".") for c in chunks)
    rejoined = " ".join(c.text for c in chunks)
    assert rejoined == " ".join(f"Sentence number {i} is here." for i in range(40))


def test_single_overlong_sentence_is_kept_whole():
    text = "## S\n" + "word " * 200 + "end."
    chunks = chunk_markdown("d", text, max_chars=50)
    assert len(chunks) == 1


def test_indexed_text_includes_title_and_section():
    chunk = chunk_markdown("doc", DOC)[1]
    assert chunk.indexed_text.startswith("Title. Section A.")


def test_corpus_loads_all_documents():
    corpus = load_corpus(CORPUS_DIR)
    assert 12 <= len(corpus) <= 15
    chunks = chunk_corpus(corpus)
    assert {c.doc_id for c in chunks} == set(corpus)
    assert len({c.chunk_id for c in chunks}) == len(chunks)
