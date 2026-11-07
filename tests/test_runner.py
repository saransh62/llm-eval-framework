"""End-to-end eval harness with FakeLLM + FakeJudgeLLM on a 5-question stratified sample."""

from app.chunking import load_corpus
from app.config import CORPUS_DIR
from app.llm import FakeJudgeLLM
from evals.cache import JudgeCache
from evals.dataset import load_dataset, sample
from evals.judge import Judge, load_judge
from evals.runner import Judges, aggregate, run


def _judges(tmp_path):
    cache = JudgeCache(tmp_path)
    llm = FakeJudgeLLM()
    return Judges(Judge(load_judge("correctness"), llm, cache), Judge(load_judge("faithfulness"), llm, cache))


def test_run_on_small_sample(pipeline, settings, tmp_path):
    items = sample(load_dataset(), 5)
    records, metrics = run(items, pipeline, _judges(tmp_path), settings, set(load_corpus(CORPUS_DIR)))
    assert [r["id"] for r in records] == [i.id for i in items]
    for key in ("retrieval_recall_at_k", "refusal_accuracy", "safety_pass_rate", "latency_p95_ms"):
        assert metrics[key] is not None
    for value in metrics.values():
        assert value is None or value >= 0
    for r in records:
        assert r["severity"] == 0 or r["failures"]


def test_unsafe_answer_is_ranked_worst():
    base = {
        "should_refuse": False,
        "refused": False,
        "latency_ms": 1,
        "input_tokens": 1,
        "output_tokens": 1,
        "answer_cost_usd": 0,
        "judge_cost_usd": 0,
    }
    records = [
        {**base, "category": "adversarial", "metrics": {"safety_pass": 0.0}},
        {**base, "category": "factual", "metrics": {"recall_at_k": 1.0, "key_fact_coverage": 0.5}},
    ]
    metrics = aggregate(records)
    assert metrics["safety_pass_rate"] == 0.0
    assert metrics["key_fact_coverage"] == 0.5
    assert metrics["refusal_accuracy"] == 1.0
