import pytest

from app.llm import LLMResponse
from evals.cache import JudgeCache, cache_key
from evals.judge import Judge, JudgeSpec, load_judge, parse_verdict


class ScriptedLLM:
    model = "scripted"

    def __init__(self, *outputs):
        self.outputs = list(outputs)
        self.calls = 0

    def complete(self, system, user):
        self.calls += 1
        return LLMResponse(self.outputs.pop(0), 10, 5, self.model)


SPEC = JudgeSpec("correctness", "correctness-v1", "rubric text")


def test_load_judge_reads_front_matter():
    spec = load_judge("correctness")
    assert spec.name == "correctness" and spec.version.startswith("correctness-v")
    assert "Rubric" in spec.rubric and "---" not in spec.rubric.splitlines()[0]
    assert load_judge("faithfulness").name == "faithfulness"
    with pytest.raises(FileNotFoundError):
        load_judge("nope")


def test_fingerprint_changes_with_rubric_text():
    assert SPEC.fingerprint != JudgeSpec("correctness", "correctness-v1", "rubric text!").fingerprint


@pytest.mark.parametrize(
    "text",
    ['{"score": 4, "reasoning": "ok"}', 'Sure!\n```json\n{"score": 4}\n```', '{"score": 4, "claims": []} trailing'],
)
def test_parse_verdict_tolerates_wrapping(text):
    assert parse_verdict(text)["score"] == 4


@pytest.mark.parametrize("text", ["no json", '{"score": 9}', '{"score": "4"}', '{"score": 3, "claims": {}}'])
def test_parse_verdict_rejects_bad_output(text):
    with pytest.raises(ValueError):
        parse_verdict(text)


def test_cache_prevents_second_call(tmp_path):
    llm = ScriptedLLM('{"score": 5, "reasoning": "all facts"}')
    judge = Judge(SPEC, llm, JudgeCache(tmp_path))
    first = judge.evaluate(question="q", reference="r", answer="a")
    second = judge.evaluate(question="q", reference="r", answer="a")
    assert first.score == second.score == 5
    assert not first.cached and second.cached
    assert llm.calls == 1
    assert second.input_tokens == 0  # cached verdicts cost nothing


def test_cache_key_depends_on_answer_and_version(tmp_path):
    llm = ScriptedLLM('{"score": 5}', '{"score": 2}', '{"score": 3}')
    cache = JudgeCache(tmp_path)
    Judge(SPEC, llm, cache).evaluate(question="q", answer="a")
    assert Judge(SPEC, llm, cache).evaluate(question="q", answer="b").score == 2
    v2 = JudgeSpec("correctness", "correctness-v2", "rubric text")
    assert Judge(v2, llm, cache).evaluate(question="q", answer="a").score == 3
    assert llm.calls == 3


def test_retry_then_error(tmp_path):
    llm = ScriptedLLM("garbage", '{"score": 3}')
    assert Judge(SPEC, llm, JudgeCache(tmp_path)).evaluate(answer="a").score == 3
    bad = ScriptedLLM("garbage", "still garbage")
    result = Judge(SPEC, bad, JudgeCache(tmp_path)).evaluate(answer="zzz")
    assert result.score is None and "unparseable" in result.error
    assert result.input_tokens == 20


def test_disabled_cache_never_hits(tmp_path):
    cache = JudgeCache(tmp_path, enabled=False)
    cache.put("k", {"score": 1})
    assert cache.get("k") is None and not any(tmp_path.iterdir())


def test_cache_key_is_stable():
    assert cache_key("a", "b") == cache_key("a", "b") != cache_key("ab")
