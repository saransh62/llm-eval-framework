"""Provider adapters tested with stubbed SDK clients (no network, no keys)."""

from types import SimpleNamespace

import pytest

from app.config import ModelConfig
from app.llm import FakeJudgeLLM, FakeLLM, make_llm
from app.prompts import REFUSAL
from app.providers import AnthropicLLM, OpenAILLM

CFG = ModelConfig("anthropic", "claude-opus-5-5", 0.0, 512, "low")


class _Recorder:
    def __init__(self, response):
        self.response, self.kwargs = response, None

    def create(self, **kwargs):
        self.kwargs = kwargs
        return self.response


def _anthropic(monkeypatch, response):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test")
    llm = AnthropicLLM(CFG)
    rec = _Recorder(response)
    llm.client = SimpleNamespace(beta=SimpleNamespace(messages=rec))
    return llm, rec


def test_anthropic_request_and_response(monkeypatch):
    resp = SimpleNamespace(
        stop_reason="end_turn",
        content=[SimpleNamespace(type="thinking"), SimpleNamespace(type="text", text=" $49 [pricing] ")],
        usage=SimpleNamespace(input_tokens=100, output_tokens=7),
        model="claude-opus-5-5",
    )
    llm, rec = _anthropic(monkeypatch, resp)
    out = llm.complete("sys", "user")
    assert out.text == "$49 [pricing]" and out.input_tokens == 100 and out.output_tokens == 7
    assert rec.kwargs["system"] == "sys" and rec.kwargs["max_tokens"] == 512
    assert rec.kwargs["output_config"] == {"effort": "low"}
    assert rec.kwargs["fallbacks"] == "default"
    assert "temperature" not in rec.kwargs


def test_anthropic_safety_refusal_maps_to_refusal_text(monkeypatch):
    resp = SimpleNamespace(stop_reason="refusal", content=[], usage=SimpleNamespace(input_tokens=5,
                           output_tokens=0), model="claude-opus-5-5")
    llm, _ = _anthropic(monkeypatch, resp)
    assert llm.complete("s", "u").text == REFUSAL


def test_openai_request_and_response(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test")
    llm = OpenAILLM(ModelConfig("openai", "gpt-4.1-mini", 0.0, 256, "low"))
    resp = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content="hi [sla]"))],
        usage=SimpleNamespace(prompt_tokens=10, completion_tokens=3),
        model="gpt-4.1-mini",
    )
    rec = _Recorder(resp)
    llm.client = SimpleNamespace(chat=SimpleNamespace(completions=rec))
    out = llm.complete("sys", "user")
    assert out.text == "hi [sla]" and out.input_tokens == 10
    assert rec.kwargs["temperature"] == 0.0
    assert rec.kwargs["messages"][0] == {"role": "system", "content": "sys"}


def test_make_llm_dispatch():
    fake = ModelConfig("fake", "fake-x", 0, 10, "low")
    assert isinstance(make_llm(fake), FakeLLM)
    assert isinstance(make_llm(fake, judge=True), FakeJudgeLLM)
    with pytest.raises(ValueError):
        make_llm(ModelConfig("nope", "m", 0, 10, "low"))
