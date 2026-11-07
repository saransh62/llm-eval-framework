"""Markdown + JSON reporting."""

from __future__ import annotations

import json
from pathlib import Path

from evals.gating import Check, checks_as_dicts, passed

REPORTS_DIR = Path(__file__).parent / "reports"
_STATUS = {"pass": "PASS", "fail": "FAIL", "skip": "n/a"}


def _fmt(value: float | None, metric: str = "") -> str:
    if value is None:
        return "-"
    if metric.endswith("_usd"):
        return f"${value:.5f}"
    if metric.endswith("_ms"):
        return f"{value:,.0f}"
    return f"{value:.3f}"


def summary_table(checks: list[Check], metrics: dict[str, float | None]) -> str:
    rows = [
        "| Metric | Value | Gate | Baseline | Status |",
        "|---|---:|---:|---:|---|",
    ]
    for c in checks:
        gate = f"{'≥' if c.direction == 'min' else '≤'} {_fmt(c.threshold, c.metric)}"
        status = _STATUS[c.status] + (f" ({'; '.join(c.reasons)})" if c.status == "fail" else "")
        rows.append(f"| {c.metric} | {_fmt(c.value, c.metric)} | {gate} | {_fmt(c.baseline, c.metric)} | {status} |")
    gated = {c.metric for c in checks}
    for name, value in metrics.items():
        if name not in gated:
            rows.append(f"| {name} | {_fmt(value, name)} | - | - | info |")
    return "\n".join(rows)


def worst_failures(records: list[dict], n: int = 5) -> list[dict]:
    failing = [r for r in records if r["severity"] > 0]
    return sorted(failing, key=lambda r: (-r["severity"], r["id"]))[:n]


def render_markdown(records: list[dict], metrics: dict, checks: list[Check], meta: dict) -> str:
    verdict = "PASSED" if passed(checks) else "FAILED"
    lines = [
        "# LLM evaluation report",
        "",
        f"**Result: {verdict}** - {meta['sample_size']} questions, provider `{meta['provider']}`, "
        f"model `{meta['model']}`, judge `{meta['judge_model']}`, {meta['timestamp']}",
        "",
        f"Judge cache: {meta['judge_cache_hits']} hits / {meta['judge_cache_misses']} misses. "
        f"Total eval cost (answers + judges): ${metrics['eval_cost_usd']:.4f}. Wall time {meta['wall_time_s']:.1f}s.",
        "",
        "## Summary",
        "",
        summary_table(checks, metrics),
        "",
        "## Worst failures",
        "",
    ]
    worst = worst_failures(records)
    if not worst:
        lines.append("No failures.")
    for r in worst:
        lines += [
            f"### {r['id']} ({r['category']}{', ' + r['attack_type'] if r['attack_type'] else ''}) "
            f"- severity {r['severity']}",
            "",
            f"**Q:** {r['question']}",
            "",
            f"**A:** {r['answer']}",
            "",
            f"Retrieved: {', '.join(r['retrieved_doc_ids']) or 'nothing'}; "
            f"expected: {', '.join(r['expected_sources']) or '-'}",
            "",
            *[f"- **{f['kind']}**: {f['reason']}" for f in r["failures"]],
            "",
        ]
    by_category: dict[str, list[int]] = {}
    for r in records:
        by_category.setdefault(r["category"], [0, 0])
        by_category[r["category"]][0] += 1
        by_category[r["category"]][1] += r["severity"] == 0
    lines += ["## Clean items by category", "", "| Category | Clean / total |", "|---|---|"]
    lines += [f"| {c} | {ok} / {total} |" for c, (total, ok) in sorted(by_category.items())]
    return "\n".join(lines) + "\n"


def write_reports(records: list[dict], metrics: dict, checks: list[Check], meta: dict, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    report = out_dir / "report.md"
    report.write_text(render_markdown(records, metrics, checks, meta))
    (out_dir / "summary.md").write_text(
        f"## LLM eval: {'PASSED' if passed(checks) else 'FAILED'}\n\n{summary_table(checks, metrics)}\n"
    )
    payload = {"meta": meta, "metrics": metrics, "checks": checks_as_dicts(checks), "records": records}
    (out_dir / "results.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    return report
