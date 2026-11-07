"""Real LLM providers, imported lazily by app.llm.make_llm so offline tests never need them configured."""

from __future__ import annotations

import anthropic
import openai

from app.config import ModelConfig
from app.llm import LLMResponse
from app.prompts import REFUSAL

# Models that accept the server-side refusal fallback (re-runs a safety-declined request on another model).
_FALLBACK_MODELS = {"claude-opus-5-5", "claude-sonnet-5-5", "claude-opus-5", "claude-fable-5-1"}


class AnthropicLLM:
    """Claude via the Messages API.

    Current Claude models reject sampling parameters (temperature/top_p), so determinism comes from a fixed
    model id, a fixed prompt and low effort rather than temperature=0. See README "Assumptions".
    """

    def __init__(self, cfg: ModelConfig):
        self.cfg = cfg
        self.model = cfg.model
        self.client = anthropic.Anthropic(max_retries=4)

    def complete(self, system: str, user: str) -> LLMResponse:
        kwargs: dict = {}
        if self.model in _FALLBACK_MODELS:
            kwargs = {"betas": ["server-side-fallback-2026-07-01"], "fallbacks": "default"}
        response = self.client.beta.messages.create(
            model=self.model,
            max_tokens=self.cfg.max_tokens,
            system=system,
            messages=[{"role": "user", "content": user}],
            output_config={"effort": self.cfg.effort},
            **kwargs,
        )
        if response.stop_reason == "refusal":
            text = REFUSAL
        else:
            text = "".join(block.text for block in response.content if block.type == "text").strip()
        return LLMResponse(
            text=text,
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
            model=response.model,
        )


class OpenAILLM:
    """OpenAI Chat Completions with temperature from config (0 by default)."""

    def __init__(self, cfg: ModelConfig):
        self.cfg = cfg
        self.model = cfg.model
        self.client = openai.OpenAI(max_retries=4)

    def complete(self, system: str, user: str) -> LLMResponse:
        response = self.client.chat.completions.create(
            model=self.model,
            temperature=self.cfg.temperature,
            max_completion_tokens=self.cfg.max_tokens,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        )
        usage = response.usage
        return LLMResponse(
            text=(response.choices[0].message.content or "").strip(),
            input_tokens=usage.prompt_tokens if usage else 0,
            output_tokens=usage.completion_tokens if usage else 0,
            model=response.model,
        )
