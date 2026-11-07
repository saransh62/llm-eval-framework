import json

import pytest

from evals.gating import gate, load_thresholds, passed, write_baseline
from evals.report import render_markdown, summary_table, worst_failures, write_reports
from evals.run import main

THRESHOLDS = {
    "default_margin": 0.05,
    "metrics": {
        "faithfulness": {"min": 0.9},
        "safety_pass_rate": {"min": 1.0, "margin": 0.0},
        "latency_p95_ms": {"max": 1000, "margin": 200},
    },
}


def _by_metric(checks):
    return {c.metric: c for c in checks}


def test_all_pass_without_baseline():
    checks = gate({"faithfulness": 0.95, "safety_pass_rate": 1.0, "latency_p95_ms": 500}, THRESHOLDS, None)
    assert passed(checks)
    assert all(c.status == "pass" for c in checks)


def test_threshold_breach_fails():
    c = _by_metric(gate({"faithfulness": 0.85, "safety_pass_rate": 1.0, "latency_p95_ms": 1500}, THRESHOLDS, None))
    assert c["faithfulness"].status == "fail" and "below min" in c["faithfulness"].reasons[0]
    assert c["latency_p95_ms"].status == "fail" and "above max" in c["latency_p95_ms"].reasons[0]


def test_regression_vs_baseline_fails_even_above_threshold():
    baseline = {"metrics": {"faithfulness": 1.0, "latency_p95_ms": 300}}
    metrics = {"faithfulness": 0.93, "safety_pass_rate": 1.0, "latency_p95_ms": 550}
    c = _by_metric(gate(metrics, THRESHOLDS, baseline))
    assert c["faithfulness"].status == "fail" and "regressed" in c["faithfulness"].reasons[0]
    assert c["latency_p95_ms"].status == "fail"  # 550 > 300 + 200
    assert not passed(list(c.values()))


def test_within_margin_passes():
    baseline = {"metrics": {"faithfulness": 0.96, "latency_p95_ms": 400}}
    c = gate({"faithfulness": 0.92, "safety_pass_rate": 1.0, "latency_p95_ms": 590}, THRESHOLDS, baseline)
    assert passed(c)


def test_zero_tolerance_metric():
    baseline = {"metrics": {"safety_pass_rate": 1.0}}
    c = _by_metric(gate({"faithfulness": 1, "safety_pass_rate": 0.99, "latency_p95_ms": 1}, THRESHOLDS, baseline))
    assert c["safety_pass_rate"].status == "fail"


def test_missing_metric_is_skipped_not_failed():
    c = _by_metric(gate({"faithfulness": None, "safety_pass_rate": 1.0, "latency_p95_ms": 1}, THRESHOLDS, None))
    assert c["faithfulness"].status == "skip"
    assert passed(list(c.values()))


def test_threshold_file_is_valid(tmp_path):
    assert "safety_pass_rate" in load_thresholds()["metrics"]
    bad = tmp_path / "t.yaml"
    bad.write_text("metrics:\n  x: {min: 1, max: 2}\n")
    with pytest.raises(ValueError):
        load_thresholds(bad)


def _record(id_, severity, kind="safety"):
    return {
        "id": id_,
        "category": "adversarial",
        "attack_type": "jailbreak",
        "question": "q",
        "answer": "a",
        "retrieved_doc_ids": [],
        "expected_sources": [],
        "severity": severity,
        "failures": [{"kind": kind, "reason": "r"}] if severity else [],
    }


def test_worst_failures_ordering_and_limit():
    records = [_record("q1", 1), _record("q2", 5), _record("q3", 0), _record("q4", 5)] + [
        _record(f"z{i}", 2) for i in range(5)
    ]
    worst = worst_failures(records)
    assert [r["id"] for r in worst[:2]] == ["q2", "q4"]
    assert len(worst) == 5 and all(r["severity"] > 0 for r in worst)


def test_report_rendering(tmp_path):
    metrics = {"faithfulness": 0.8, "safety_pass_rate": 1.0, "latency_p95_ms": 10, "eval_cost_usd": 0.01}
    checks = gate(metrics, THRESHOLDS, None)
    meta = {
        "sample_size": 1,
        "provider": "fake",
        "model": "m",
        "judge_model": "j",
        "timestamp": "t",
        "judge_cache_hits": 0,
        "judge_cache_misses": 1,
        "wall_time_s": 0.1,
    }
    md = render_markdown([_record("q1", 5)], metrics, checks, meta)
    assert "**Result: FAILED**" in md and "### q1" in md and "| faithfulness | 0.800 |" in md
    assert "| eval_cost_usd | $0.01000 | - | - | info |" in summary_table(checks, metrics)
    write_reports([_record("q1", 5)], metrics, checks, meta, tmp_path)
    assert json.loads((tmp_path / "results.json").read_text())["records"][0]["id"] == "q1"
    assert (tmp_path / "summary.md").read_text().startswith("## LLM eval: FAILED")


def test_write_baseline(tmp_path):
    path = tmp_path / "b.json"
    write_baseline({"faithfulness": 0.9}, {"model": "m"}, path)
    assert json.loads(path.read_text()) == {"meta": {"model": "m"}, "metrics": {"faithfulness": 0.9}}


def test_cli_end_to_end_with_fake_llm(tmp_path, monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "fake")  # main() mutates os.environ; monkeypatch restores it
    baseline = tmp_path / "baseline.json"
    args = [
        "--provider",
        "fake",
        "--sample",
        "5",
        "--out",
        str(tmp_path / "out"),
        "--cache-dir",
        str(tmp_path / "cache"),
        "--baseline",
        str(baseline),
    ]
    assert main([*args, "--update-baseline", "--no-gate"]) == 0
    assert baseline.exists() and (tmp_path / "out" / "report.md").exists()
    # Second run: identical fake outputs, so every judge verdict comes from the cache.
    main([*args, "--no-gate"])
    meta = json.loads((tmp_path / "out" / "results.json").read_text())["meta"]
    assert meta["judge_cache_misses"] == 0 and meta["judge_cache_hits"] > 0


def test_cli_refuses_real_provider_without_key(monkeypatch, tmp_path):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("JUDGE_PROVIDER", raising=False)
    monkeypatch.setenv("LLM_PROVIDER", "fake")
    assert main(["--provider", "anthropic", "--out", str(tmp_path)]) == 2
