"""Split markdown documents into heading-aware chunks."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    doc_id: str
    title: str
    section: str
    text: str

    @property
    def indexed_text(self) -> str:
        """Text used for retrieval: title and section names carry useful keywords."""
        return f"{self.title}. {self.section}. {self.text}"


def load_corpus(corpus_dir: Path) -> dict[str, str]:
    """Return {doc_id: markdown} for every .md file; doc_id is the file stem."""
    return {p.stem: p.read_text(encoding="utf-8") for p in sorted(corpus_dir.glob("*.md"))}


def _split_long(text: str, max_chars: int) -> list[str]:
    """Greedily pack sentences into pieces of at most max_chars (a single long sentence stays whole)."""
    pieces: list[str] = []
    current = ""
    for sentence in _SENTENCE_SPLIT.split(text):
        candidate = f"{current} {sentence}".strip()
        if current and len(candidate) > max_chars:
            pieces.append(current)
            current = sentence
        else:
            current = candidate
    if current:
        pieces.append(current)
    return pieces


def chunk_markdown(doc_id: str, markdown: str, max_chars: int = 600) -> list[Chunk]:
    """Chunk by '## ' sections; sections longer than max_chars are split on sentence boundaries."""
    title = doc_id
    sections: list[tuple[str, list[str]]] = []
    current_section: str | None = None
    for line in markdown.splitlines():
        if line.startswith("# "):
            title = line[2:].strip()
            current_section = title
            sections.append((current_section, []))
        elif line.startswith("## "):
            current_section = line[3:].strip()
            sections.append((current_section, []))
        else:
            if current_section is None:
                current_section = title
                sections.append((current_section, []))
            sections[-1][1].append(line)

    chunks: list[Chunk] = []
    for section, lines in sections:
        body = " ".join(line.strip() for line in lines if line.strip())
        if not body:
            continue
        for piece in _split_long(body, max_chars):
            chunks.append(Chunk(f"{doc_id}#{len(chunks)}", doc_id, title, section, piece))
    return chunks


def chunk_corpus(corpus: dict[str, str], max_chars: int = 600) -> list[Chunk]:
    return [c for doc_id, md in corpus.items() for c in chunk_markdown(doc_id, md, max_chars)]
