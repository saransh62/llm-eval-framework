"""FastAPI service: POST /ask answers questions about Acme Cloud from its documentation."""

from __future__ import annotations

from typing import Annotated

from fastapi import Depends, FastAPI
from pydantic import BaseModel, Field

from app.rag import RAGPipeline, default_pipeline


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)


class Source(BaseModel):
    doc_id: str
    chunk_id: str
    score: float


class Tokens(BaseModel):
    input: int
    output: int
    total: int


class AskResponse(BaseModel):
    answer: str
    sources: list[Source]
    citations: list[str]
    refused: bool
    latency_ms: float
    tokens: Tokens
    model: str


def get_pipeline() -> RAGPipeline:
    return default_pipeline()


app = FastAPI(title="Acme Cloud Docs Q&A", version="0.1.0")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest, pipeline: Annotated[RAGPipeline, Depends(get_pipeline)]) -> AskResponse:
    result = pipeline.ask(req.question.strip())
    return AskResponse(
        answer=result.answer,
        sources=[Source(**s) for s in result.sources],
        citations=result.citations,
        refused=result.refused,
        latency_ms=result.latency_ms,
        tokens=Tokens(
            input=result.input_tokens,
            output=result.output_tokens,
            total=result.input_tokens + result.output_tokens,
        ),
        model=result.model,
    )
