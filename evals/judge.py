"""LLM-as-judge with versioned rubric files, robust JSON parsing and result caching."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from html import escape
from pathlib import Path

from app.llm import LLMClient
from evals.cache import JudgeCache, cache_key

JUDGES_DIR = Path(__file__).parent / "judges"
_FRONT_MATTER = re.compile(r"\A---\n(.*?)\n---\n(.*)\Z", re.S)


@dataclass(frozen=True)
class JudgeSpec:
    name: str
    version: str
    rubric: str

    @property
    def fingerprint(self) -> str:
        """Version label plus a content hash: editing a rubric without bumping the version still busts the cache."""
        return f"{self.version}:{hashlib.sha256(self.rubric.encode()).hexdigest()[:12]}"


def load_judge(name: str, directory: Path = JUDGES_DIR) -> JudgeSpec:
    """Load the highest-versioned rubric file for a judge, e.g. correctness_v2.md over correctness_v1.md."""
    candidates = sorted(directory.glob(f"{name}_v*.md"), key=lambda p: int(p.stem.rsplit("_v", 1)[1]))
    if not candidates:
        raise FileNotFoundError(f"no judge rubric named {name}_v*.md in {directory}")
    match = _FRONT_MATTER.match(candidates[-1].read_text())
    if not match:
        raise ValueError(f"{candidates[-1]} is missing its --- front matter ---")
    meta = dict(line.split(":", 1) for line in match.group(1).splitlines() if ":" in line)
    return JudgeSpec(meta["name"].strip(), meta["version"].strip(), match.group(2).strip())


@dataclass
class JudgeResult:
    score: int | None
    reasoning: str = ""
    claims: list[dict] = field(default_factory=list)
    cached: bool = False
    input_tokens: int = 0
    output_tokens: int = 0
    error: str | None = None


def parse_verdict(text: str) -> dict:
    """Extract the first JSON object from the judge output (tolerates code fences or stray prose)."""
    start = text.find("{")
    if start < 0:
        raise ValueError("no JSON object in judge output")
    obj, _ = json.JSONDecoder().raw_decode(text[start:])
    score = obj.get("score")
    if not isinstance(score, int) or not 1 <= score <= 5:
        raise ValueError(f"score must be an integer 1-5, got {score!r}")
    claims = obj.get("claims", [])
    if not isinstance(claims, list):
        raise ValueError("claims must be a list")
    return obj


class Judge:
    def __init__(self, spec: JudgeSpec, llm: LLMClient, cache: JudgeCache, max_attempts: int = 2):
        self.spec = spec
        self.llm = llm
        self.cache = cache
        self.max_attempts = max_attempts

    def evaluate(self, **fields: str) -> JudgeResult:
        """fields become tagged sections of the judge's user message, e.g. question=..., answer=..."""
        key = cache_key(self.spec.fingerprint, self.llm.model, *(f"{k}={fields[k]}" for k in sorted(fields)))
        hit = self.cache.get(key)
        if hit is not None:
            return JudgeResult(hit["score"], hit.get("reasoning", ""), hit.get("claims", []), cached=True)

        user = "\n\n".join(f"<{k}>\n{escape(v, quote=False)}\n</{k}>" for k, v in fields.items())
        tokens_in = tokens_out = 0
        error = "judge was not called"
        for _ in range(self.max_attempts):
            response = self.llm.complete(self.spec.rubric, user)
            tokens_in += response.input_tokens
            tokens_out += response.output_tokens
            try:
                verdict = parse_verdict(response.text)
            except (ValueError, json.JSONDecodeError) as exc:
                error = f"unparseable judge output: {exc}"
                continue
            result = {
                "score": verdict["score"],
                "reasoning": str(verdict.get("reasoning", "")),
                "claims": verdict.get("claims", []),
            }
            self.cache.put(key, result)
            return JudgeResult(**result, input_tokens=tokens_in, output_tokens=tokens_out)
        return JudgeResult(None, error=error, input_tokens=tokens_in, output_tokens=tokens_out)
