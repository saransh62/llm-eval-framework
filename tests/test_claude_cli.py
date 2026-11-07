"""ClaudeCLILLM tested with a stubbed subprocess.run (no claude binary, no network)."""

import json
import subprocess
from types import SimpleNamespace

import pytest

from app import claude_cli
from app.claude_cli import ClaudeCLIError, ClaudeCLILLM
from app.config import ModelConfig
from app.llm import make_llm
from app.prompts import REFUSAL

CFG = ModelConfig("claude-cli", "claude-sonnet-5-5", 0.0, 512, "low")


def _result(**overrides):
    data = {
        "is_error": False,
        "result": " $49 per month [pricing]. ",
        "stop_reason": "end_turn",
        "usage": {
            "input_tokens": 10,
            "cache_creation_input_tokens": 4000,
            "cache_read_input_tokens": 300,
            "output_tokens": 12,
        },
        "modelUsage": {"claude-sonnet-5-5": {}},
    }
    data.update(overrides)
    return json.dumps(data)


def _stub(monkeypatch, outputs):
    calls = []
    outputs = list(outputs)

    def fake_run(cmd, **kwargs):
        calls.append((cmd, kwargs))
        return SimpleNamespace(stdout=outputs.pop(0), stderr="", returncode=0)

    monkeypatch.setattr(claude_cli.subprocess, "run", fake_run)
    monkeypatch.setattr(claude_cli.time, "sleep", lambda s: None)
    return calls


def test_request_shape_and_response(monkeypatch):
    monkeypatch.setenv("CLAUDECODE", "1")
    monkeypatch.setenv("CLAUDE_CODE_SESSION_ID", "parent")
    monkeypatch.setenv("ANTHROPIC_BASE_URL", "http://parent-proxy")
    calls = _stub(monkeypatch, [_result()])
    out = ClaudeCLILLM(CFG).complete("SYSTEM", "USER")

    cmd, kwargs = calls[0]
    assert cmd[:2] == ["claude", "-p"]
    assert cmd[cmd.index("--system-prompt") + 1] == "SYSTEM"
    assert cmd[cmd.index("--model") + 1] == "claude-sonnet-5-5"
    assert cmd[cmd.index("--tools") + 1] == ""
    assert cmd[cmd.index("--effort") + 1] == "low"
    assert {"--strict-mcp-config", "--disable-slash-commands", "--no-session-persistence"} <= set(cmd)
    assert kwargs["input"] == "USER"
    env = kwargs["env"]
    assert "CLAUDECODE" not in env and "CLAUDE_CODE_SESSION_ID" not in env and "ANTHROPIC_BASE_URL" not in env

    assert out.text == "$49 per month [pricing]."
    assert out.input_tokens == 4310
    assert out.output_tokens == 12
    assert out.model == "claude-sonnet-5-5"


def test_refusal_stop_reason_maps_to_refusal_text(monkeypatch):
    _stub(monkeypatch, [_result(stop_reason="refusal", result="")])
    assert ClaudeCLILLM(CFG).complete("s", "u").text == REFUSAL


def test_retries_then_succeeds(monkeypatch):
    calls = _stub(monkeypatch, [_result(is_error=True, result="overloaded"), "not json", _result()])
    assert ClaudeCLILLM(CFG, retries=2).complete("s", "u").output_tokens == 12
    assert len(calls) == 3


def test_raises_after_retries(monkeypatch):
    _stub(monkeypatch, [_result(is_error=True, result="Not logged in")] * 2)
    with pytest.raises(ClaudeCLIError, match="Not logged in"):
        ClaudeCLILLM(CFG, retries=1).complete("s", "u")


def test_timeout_is_retried(monkeypatch):
    outputs = iter([subprocess.TimeoutExpired("claude", 1), _result()])

    def fake_run(cmd, **kwargs):
        item = next(outputs)
        if isinstance(item, Exception):
            raise item
        return SimpleNamespace(stdout=item, stderr="", returncode=0)

    monkeypatch.setattr(claude_cli.subprocess, "run", fake_run)
    monkeypatch.setattr(claude_cli.time, "sleep", lambda s: None)
    assert ClaudeCLILLM(CFG).complete("s", "u").text.startswith("$49")


def test_make_llm_builds_cli_client():
    assert isinstance(make_llm(CFG), ClaudeCLILLM)
