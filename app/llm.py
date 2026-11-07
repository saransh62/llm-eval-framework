"""Provider-agnostic LLM client plus deterministic fakes for offline tests."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from html import unescape
from typing import Protocol

from app.config import ModelConfig
from app.prompts import REFUSAL
from app.retrieval import tokenize


@dataclass(frozen=True)
class LLMResponse:
    text: str
    input_tokens: int
    output_tokens: int
    model: str


class LLMClient(Protocol):
    model: str

    def complete(self, system: str, user: str) -> LLMResponse: ...


def _estimate_tokens(text: str) -> int:
    return max(1, round(len(text) / 4))


_DOC_BLOCK = re.compile(r'<doc id="([^"]+)"[^>]*>\n(.*?)\n</doc>', re.S)
_TAG = re.compile(r"<(question|answer|reference|context)>(.*?)</\1>", re.S)
_SENTENCES = re.compile(r"(?<=[.!?])\s+")


def _overlap(query_tokens: set[str], text: str) -> float:
    if not query_tokens:
        return 0.0
    return len(query_tokens & set(tokenize(text))) / len(query_tokens)


class FakeLLM:
    """Deterministic extractive 'LLM': returns the context sentence that best overlaps the question.

    It refuses when no sentence covers enough of the question. It is not meant to be smart; it exists so the
    API, pipeline and eval harness can be tested end-to-end without network access or cost.
    """

    def __init__(self, model: str = "fake-extractive-v1", min_overlap: float = 0.5):
        self.model = model
        self.min_overlap = min_overlap

    def complete(self, system: str, user: str) -> LLMResponse:
        tags = dict(_TAG.findall(user))
        question_tokens = set(tokenize(unescape(tags.get("question", ""))))
        best: tuple[float, str, str] = (0.0, "", "")
        for doc_id, body in _DOC_BLOCK.findall(user):
            for sentence in _SENTENCES.split(unescape(body)):
                score = _overlap(question_tokens, sentence)
                if score > best[0]:
                    best = (score, sentence.strip(), doc_id)
        if best[0] < self.min_overlap:
            text = REFUSAL
        else:
            text = f"{best[1].rstrip('.')} [{best[2]}]."
        return LLMResponse(text, _estimate_tokens(system + user), _estimate_tokens(text), self.model)


class FakeJudgeLLM:
    """Deterministic stand-in for the judge. Reads the <answer>, <reference> and <context> tags of a judge
    prompt and returns JSON that satisfies every judge's schema (score + per-claim support)."""

    def __init__(self, model: str = "fake-judge-v1"):
        self.model = model

    def complete(self, system: str, user: str) -> LLMResponse:
        tags = dict(_TAG.findall(user))
        answer = unescape(tags.get("answer", ""))
        reference = unescape(tags.get("reference", ""))
        context = unescape(tags.get("context", ""))
        answer_tokens = set(tokenize(answer))
        coverage = _overlap(set(tokenize(reference)), answer) if reference else 0.0
        score = 1 + round(4 * coverage)
        claims = []
        context_tokens = set(tokenize(context))
        for sentence in filter(None, (s.strip() for s in _SENTENCES.split(answer))):
            tokens = set(tokenize(re.sub(r"\[[a-z0-9-]+\]", "", sentence)))
            if not tokens:
                continue
            supported = bool(tokens) and len(tokens & context_tokens) / len(tokens) >= 0.6
            claims.append({"claim": sentence, "supported": supported})
        verdict = {
            "score": score,
            "reasoning": f"fake judge: reference coverage {coverage:.2f}, {len(answer_tokens)} answer tokens",
            "claims": claims,
        }
        text = json.dumps(verdict)
        return LLMResponse(text, _estimate_tokens(system + user), _estimate_tokens(text), self.model)


def make_llm(cfg: ModelConfig, *, judge: bool = False) -> LLMClient:
    """Build a client for cfg.provider. Real providers are imported lazily, so offline runs need no keys."""
    if cfg.provider == "fake":
        return FakeJudgeLLM(cfg.model) if judge else FakeLLM(cfg.model)
    if cfg.provider == "anthropic":
        from app.providers import AnthropicLLM

        return AnthropicLLM(cfg)
    if cfg.provider == "openai":
        from app.providers import OpenAILLM

        return OpenAILLM(cfg)
    raise ValueError(f"Unknown LLM provider: {cfg.provider!r} (expected fake, anthropic or openai)")
