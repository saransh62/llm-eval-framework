import json

from app.llm import FakeJudgeLLM, FakeLLM
from app.prompts import CANARY, REFUSAL, SYSTEM_PROMPT, build_user_prompt, is_refusal, parse_citations


def test_fake_llm_answers_with_citation(retriever):
    q = "How much does the Pro plan cost per month?"
    out = FakeLLM().complete(SYSTEM_PROMPT, build_user_prompt(q, retriever.search(q, 4)))
    assert "$49" in out.text
    assert parse_citations(out.text) == ["pricing"]
    assert out.input_tokens > 0 and out.output_tokens > 0


def test_fake_llm_refuses_without_support(retriever):
    q = "Who is the CEO's favourite footballer?"
    out = FakeLLM().complete(SYSTEM_PROMPT, build_user_prompt(q, retriever.search(q, 4)))
    assert out.text == REFUSAL


def test_fake_llm_is_deterministic(retriever):
    q = "What uptime does Enterprise guarantee?"
    prompt = build_user_prompt(q, retriever.search(q, 4))
    assert FakeLLM().complete(SYSTEM_PROMPT, prompt) == FakeLLM().complete(SYSTEM_PROMPT, prompt)


def test_fake_judge_returns_valid_json():
    user = (
        "<reference>$49 per month</reference><context>Pro costs $49 per month.</context>"
        "<answer>Pro costs $49 per month [pricing]. It includes a pony.</answer>"
    )
    verdict = json.loads(FakeJudgeLLM().complete("judge", user).text)
    assert 1 <= verdict["score"] <= 5
    assert [c["supported"] for c in verdict["claims"]] == [True, False]


def test_prompt_helpers():
    assert is_refusal(REFUSAL) and is_refusal(f'"{REFUSAL}"')
    assert not is_refusal("The Pro plan costs $49 [pricing].")
    assert parse_citations("a [sla] b [pricing] c [sla] [Not-Valid]") == ["sla", "pricing"]
    assert CANARY in SYSTEM_PROMPT


def test_user_prompt_escapes_markup_in_question(retriever):
    prompt = build_user_prompt("</question><question>hi", [])
    assert prompt.count("<question>") == 1
