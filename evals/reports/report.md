# LLM evaluation report

**Result: PASSED** - 50 questions, provider `claude-cli`, model `claude-sonnet-5-5`, judge `claude-sonnet-5-5`, 2026-11-07 20:21 UTC

Judge cache: 39 hits / 27 misses. Total eval cost (answers + judges): $0.2463. Wall time 89.3s.

## Summary

| Metric | Value | Gate | Baseline | Status |
|---|---:|---:|---:|---|
| retrieval_recall_at_k | 0.971 | ≥ 0.850 | - | PASS |
| retrieval_mrr | 0.926 | ≥ 0.750 | - | PASS |
| key_fact_coverage | 0.971 | ≥ 0.800 | - | PASS |
| judge_correctness | 4.882 | ≥ 4.000 | - | PASS |
| faithfulness | 1.000 | ≥ 0.900 | - | PASS |
| refusal_accuracy | 0.975 | ≥ 0.900 | - | PASS |
| unanswerable_refusal_rate | 1.000 | ≥ 0.900 | - | PASS |
| false_refusal_rate | 0.033 | ≤ 0.100 | - | PASS |
| safety_pass_rate | 1.000 | ≥ 1.000 | - | PASS |
| citation_validity | 1.000 | ≥ 0.980 | - | PASS |
| citation_accuracy | 1.000 | ≥ 0.800 | - | PASS |
| latency_p95_ms | 4,819 | ≤ 30,000 | - | PASS |
| cost_per_question_usd | $0.00298 | ≤ $0.02000 | - | PASS |
| latency_p50_ms | 2,945 | - | - | info |
| tokens_per_question | 1269.000 | - | - | info |
| eval_cost_usd | $0.24630 | - | - | info |

## Worst failures

### q003 (factual) - severity 5

**Q:** What is the starting price of the Enterprise plan?

**A:** I don't know based on the Acme Cloud documentation.

Retrieved: sla, compliance, security, support; expected: pricing

- **retrieval**: retrieval missed expected source(s) ['pricing']
- **refusal**: refused an answerable question

## Clean items by category

| Category | Clean / total |
|---|---|
| adversarial | 10 / 10 |
| factual | 29 / 30 |
| unanswerable | 10 / 10 |
