"""Runs the golden dataset through the RAG pipeline, scores every item and aggregates the metrics."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

from app.config import Settings
from app.rag import RAGPipeline
from evals.dataset import GoldenItem
from evals.judge import Judge
from evals.metrics import citation, correctness, cost, faithfulness, refusal, retrieval, safety
from evals.metrics.stats import mean, percentile

# How much each kind of failure counts when ranking the worst failures in the report.
SEVERITY = {
    "safety": 5,
    "refusal": 4,
    "faithfulness": 3,
    "correctness": 3,
    "judge_error": 2,
    "citation": 1,
    "retrieval": 1,
}


@dataclass
class Judges:
    correctness: Judge
    faithfulness: Judge


def evaluate_item(
    item: GoldenItem, pipeline: RAGPipeline, judges: Judges, settings: Settings, known_ids: set[str]
) -> dict:
    """Ask one question and score it on every applicable metric. Returns a JSON-serialisable record."""
    res = pipeline.ask(item.question)
    failures: list[tuple[str, str]] = []
    m: dict[str, float | None] = {}

    # 1. retrieval
    retrieved = res.retrieved_doc_ids
    m["recall_at_k"] = retrieval.recall_at_k(retrieved, item.expected_sources, pipeline.top_k)
    m["reciprocal_rank"] = retrieval.reciprocal_rank(retrieved, item.expected_sources)
    if m["recall_at_k"] is not None and m["recall_at_k"] < 1:
        missed = sorted(set(item.expected_sources) - set(retrieved))
        failures.append(("retrieval", f"retrieval missed expected source(s) {missed}"))

    # 4. refusal decision (factual + unanswerable; adversarial refusals are scored by safety)
    if item.category != "adversarial":
        if item.should_refuse and not res.refused:
            failures.append(("refusal", "answered an unanswerable question (hallucination risk)"))
        if not item.should_refuse and res.refused:
            failures.append(("refusal", "refused an answerable question"))

    # 2. correctness: key facts + judge
    judge_cost_in = judge_cost_out = 0
    m["judge_correctness"] = None
    if item.answerable and item.expected_facts:
        m["key_fact_coverage"] = correctness.key_fact_coverage(item.expected_facts, res.answer)
        missing = correctness.missing_facts(item.expected_facts, res.answer)
        if missing and not res.refused:
            failures.append(("correctness", f"missing key facts {missing}"))
        verdict = judges.correctness.evaluate(
            question=item.question, reference="; ".join(item.expected_facts), answer=res.answer
        )
        judge_cost_in += verdict.input_tokens
        judge_cost_out += verdict.output_tokens
        m["judge_correctness"] = verdict.score
        if verdict.error:
            failures.append(("judge_error", verdict.error))
        elif verdict.score is not None and verdict.score <= 2 and not res.refused:
            failures.append(("correctness", f"judge scored correctness {verdict.score}/5: {verdict.reasoning}"))
    else:
        m["key_fact_coverage"] = None

    # 3. faithfulness (only meaningful when the system actually answered)
    m["faithfulness"] = None
    if not res.refused:
        verdict = judges.faithfulness.evaluate(context=res.context_text, answer=res.answer)
        judge_cost_in += verdict.input_tokens
        judge_cost_out += verdict.output_tokens
        if verdict.error:
            failures.append(("judge_error", verdict.error))
        else:
            m["faithfulness"] = faithfulness.supported_fraction(verdict.claims)
            if m["faithfulness"] is None:
                m["faithfulness"] = 1.0 if verdict.score >= 4 else 0.0
            for claim in faithfulness.unsupported_claims(verdict.claims):
                failures.append(("faithfulness", f"unsupported claim: {claim}"))

    # 5. safety
    m["safety_pass"] = None
    if item.category == "adversarial":
        violations = safety.safety_violations(res.answer, item.forbidden, item.should_refuse, res.refused)
        m["safety_pass"] = 0.0 if violations else 1.0
        failures.extend(("safety", v) for v in violations)

    # 6. citations
    m["citation_validity"] = m["citation_accuracy"] = None
    if not res.refused:
        scores = citation.citation_scores(res.citations, known_ids, item.expected_sources)
        m["citation_validity"], m["citation_accuracy"] = scores["validity"], scores["accuracy"]
        unknown = [c for c in res.citations if c not in known_ids]
        if unknown:
            failures.append(("citation", f"cited non-existent document(s) {unknown}"))
        elif scores["accuracy"] is not None and scores["accuracy"] < 1:
            detail = "no citation" if not res.citations else f"cited irrelevant doc(s) {res.citations}"
            failures.append(("citation", detail))

    # 7. latency and cost
    answer_cost = cost.cost_usd(res.input_tokens, res.output_tokens, settings.price(settings.llm.model))
    judge_cost = cost.cost_usd(judge_cost_in, judge_cost_out, settings.price(settings.judge.model))

    return {
        "id": item.id,
        "category": item.category,
        "attack_type": item.attack_type,
        "question": item.question,
        "answer": res.answer,
        "refused": res.refused,
        "should_refuse": item.should_refuse,
        "citations": res.citations,
        "retrieved_doc_ids": retrieved,
        "expected_sources": item.expected_sources,
        "latency_ms": res.latency_ms,
        "input_tokens": res.input_tokens,
        "output_tokens": res.output_tokens,
        "answer_cost_usd": answer_cost,
        "judge_cost_usd": judge_cost,
        "metrics": m,
        "failures": [{"kind": k, "reason": r} for k, r in failures],
        "severity": sum(SEVERITY[k] for k, _ in failures),
    }


def aggregate(records: list[dict]) -> dict[str, float | None]:
    """Turn per-item records into the run-level metrics that are gated in CI."""

    def col(name: str) -> list[float | None]:
        return [r["metrics"].get(name) for r in records]

    decisions = [(r["should_refuse"], r["refused"]) for r in records if r["category"] != "adversarial"]
    n = len(records) or 1
    return {
        "retrieval_recall_at_k": mean(col("recall_at_k")),
        "retrieval_mrr": mean(col("reciprocal_rank")),
        "key_fact_coverage": mean(col("key_fact_coverage")),
        "judge_correctness": mean(col("judge_correctness")),
        "faithfulness": mean(col("faithfulness")),
        **refusal.refusal_metrics(decisions),
        "safety_pass_rate": mean(col("safety_pass")),
        "citation_validity": mean(col("citation_validity")),
        "citation_accuracy": mean(col("citation_accuracy")),
        "latency_p50_ms": percentile([r["latency_ms"] for r in records], 50),
        "latency_p95_ms": percentile([r["latency_ms"] for r in records], 95),
        "tokens_per_question": sum(r["input_tokens"] + r["output_tokens"] for r in records) / n,
        "cost_per_question_usd": sum(r["answer_cost_usd"] for r in records) / n,
        "eval_cost_usd": sum(r["answer_cost_usd"] + r["judge_cost_usd"] for r in records),
    }


def run(
    items: list[GoldenItem],
    pipeline: RAGPipeline,
    judges: Judges,
    settings: Settings,
    known_ids: set[str],
    workers: int = 4,
) -> tuple[list[dict], dict[str, float | None]]:
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        records = list(pool.map(lambda i: evaluate_item(i, pipeline, judges, settings, known_ids), items))
    return records, aggregate(records)
