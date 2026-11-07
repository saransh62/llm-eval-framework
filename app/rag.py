"""Retrieval-augmented question answering pipeline."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from functools import lru_cache

from app.chunking import chunk_corpus, load_corpus
from app.config import CORPUS_DIR, Settings, get_settings
from app.llm import LLMClient, make_llm
from app.prompts import SYSTEM_PROMPT, build_user_prompt, is_refusal, parse_citations
from app.retrieval import BM25Retriever, ScoredChunk, unique_doc_ids


@dataclass
class AskResult:
    answer: str
    sources: list[dict]
    citations: list[str]
    refused: bool
    latency_ms: float
    input_tokens: int
    output_tokens: int
    model: str
    retrieved: list[ScoredChunk] = field(repr=False, default_factory=list)

    @property
    def retrieved_doc_ids(self) -> list[str]:
        return unique_doc_ids(self.retrieved)

    @property
    def context_text(self) -> str:
        return "\n\n".join(f"[{r.chunk.doc_id}] {r.chunk.text}" for r in self.retrieved)


class RAGPipeline:
    def __init__(self, retriever: BM25Retriever, llm: LLMClient, top_k: int = 4):
        self.retriever = retriever
        self.llm = llm
        self.top_k = top_k

    def ask(self, question: str) -> AskResult:
        start = time.perf_counter()
        retrieved = self.retriever.search(question, self.top_k)
        response = self.llm.complete(SYSTEM_PROMPT, build_user_prompt(question, retrieved))
        latency_ms = (time.perf_counter() - start) * 1000
        answer = response.text.strip()
        return AskResult(
            answer=answer,
            sources=[
                {"doc_id": r.chunk.doc_id, "chunk_id": r.chunk.chunk_id, "score": round(r.score, 4)} for r in retrieved
            ],
            citations=parse_citations(answer),
            refused=is_refusal(answer),
            latency_ms=round(latency_ms, 1),
            input_tokens=response.input_tokens,
            output_tokens=response.output_tokens,
            model=response.model,
            retrieved=retrieved,
        )


def build_retriever(settings: Settings) -> BM25Retriever:
    return BM25Retriever(chunk_corpus(load_corpus(CORPUS_DIR), settings.chunk_max_chars))


def build_pipeline(settings: Settings | None = None, llm: LLMClient | None = None) -> RAGPipeline:
    settings = settings or get_settings()
    return RAGPipeline(build_retriever(settings), llm or make_llm(settings.llm), settings.top_k)


@lru_cache(maxsize=1)
def default_pipeline() -> RAGPipeline:
    return build_pipeline()
