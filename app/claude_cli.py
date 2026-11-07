"""Claude via the local Claude Code CLI (`claude -p`), so an eval can run on a Claude subscription without an API key.

Each call starts a fresh, stateless `claude` process with the app's own system prompt, every tool, MCP server and
slash command turned off, and JSON output. Latency therefore includes process start-up and is not comparable with
the API providers. See README "Assumptions".
"""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
import time

from app.config import ModelConfig
from app.llm import LLMResponse
from app.prompts import REFUSAL

# Variables a parent Claude Code session sets for its own child processes. A nested `claude` must not inherit
# them, or it tries to authenticate through the parent instead of its own sign-in.
_PARENT_ENV_PREFIXES = ("CLAUDE_CODE_", "CLAUDE_AGENT_SDK_")
_PARENT_ENV_NAMES = {"CLAUDECODE", "CLAUDE_PID", "CLAUDE_EFFORT", "ANTHROPIC_BASE_URL"}


class ClaudeCLIError(RuntimeError):
    pass


def _child_env() -> dict[str, str]:
    return {
        k: v for k, v in os.environ.items() if k not in _PARENT_ENV_NAMES and not k.startswith(_PARENT_ENV_PREFIXES)
    }


class ClaudeCLILLM:
    """Runs one `claude -p` process per completion. `CLAUDE_CLI` overrides the executable path."""

    def __init__(self, cfg: ModelConfig, executable: str | None = None, timeout_s: float = 300, retries: int = 2):
        self.cfg = cfg
        self.model = cfg.model
        self.executable = executable or os.getenv("CLAUDE_CLI", "claude")
        self.timeout_s = timeout_s
        self.retries = retries
        # Run from an empty directory so no project CLAUDE.md or settings leak into the prompt.
        self._cwd = tempfile.mkdtemp(prefix="claude-cli-eval-")

    def command(self, system: str) -> list[str]:
        cmd = [
            self.executable,
            "-p",
            "--model",
            self.model,
            "--system-prompt",
            system,
            "--tools",
            "",
            "--strict-mcp-config",
            "--disable-slash-commands",
            "--no-session-persistence",
            "--output-format",
            "json",
        ]
        if self.cfg.effort:
            cmd += ["--effort", self.cfg.effort]
        return cmd

    def _run_once(self, system: str, user: str) -> dict:
        proc = subprocess.run(
            self.command(system),
            input=user,
            capture_output=True,
            text=True,
            timeout=self.timeout_s,
            cwd=self._cwd,
            env=_child_env(),
        )
        try:
            data = json.loads(proc.stdout)
        except json.JSONDecodeError as exc:
            detail = (proc.stderr or proc.stdout).strip()[:300]
            raise ClaudeCLIError(f"claude exited {proc.returncode} without JSON output: {detail}") from exc
        if data.get("is_error"):
            raise ClaudeCLIError(f"claude reported an error: {str(data.get('result'))[:300]}")
        return data

    def complete(self, system: str, user: str) -> LLMResponse:
        for attempt in range(self.retries + 1):
            try:
                data = self._run_once(system, user)
                break
            except (ClaudeCLIError, subprocess.TimeoutExpired):
                if attempt == self.retries:
                    raise
                time.sleep(5 * (attempt + 1))
        usage = data.get("usage") or {}
        # The CLI adds its own fixed preamble, which it caches. Count every input token the request consumed.
        input_tokens = sum(
            int(usage.get(k) or 0) for k in ("input_tokens", "cache_creation_input_tokens", "cache_read_input_tokens")
        )
        if data.get("stop_reason") == "refusal":
            text = REFUSAL
        else:
            text = str(data.get("result") or "").strip()
        models = list(data.get("modelUsage") or {})
        return LLMResponse(
            text=text,
            input_tokens=input_tokens,
            output_tokens=int(usage.get("output_tokens") or 0),
            model=models[0] if len(models) == 1 else self.model,
        )
