"""CLI: run the evaluation suite, write reports, gate on thresholds/baseline.

python -m evals.run --sample 5               # FakeLLM unless LLM_PROVIDER is set
LLM_PROVIDER=anthropic python -m evals.run   # full run against a real model
"""

from __future__ import annotations

import argparse
import hashlib
import os
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

from app.chunking import load_corpus
from app.config import CORPUS_DIR, load_settings
from app.llm import make_llm
from app.prompts import PROMPT_VERSION
from app.rag import build_pipeline
from evals.cache import DEFAULT_CACHE_DIR, JudgeCache
from evals.dataset import DATASET_PATH, load_dataset, sample
from evals.gating import BASELINE_PATH, THRESHOLDS_PATH, gate, load_baseline, load_thresholds, passed, write_baseline
from evals.judge import Judge, load_judge
from evals.report import REPORTS_DIR, summary_table, write_reports
from evals.runner import Judges, run

_KEY_ENV = {"anthropic": "ANTHROPIC_API_KEY", "openai": "OPENAI_API_KEY"}


def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    env_sample = os.getenv("EVAL_SAMPLE")
    p.add_argument("--sample", type=int, default=int(env_sample) if env_sample else None, help="questions to run")
    p.add_argument("--seed", type=int, default=7, help="sampling seed")
    p.add_argument("--provider", choices=["fake", "anthropic", "openai"], help="overrides LLM_PROVIDER")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--out", type=Path, default=REPORTS_DIR)
    p.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    p.add_argument("--no-cache", action="store_true", help="ignore and do not write judge cache")
    p.add_argument("--thresholds", type=Path, default=THRESHOLDS_PATH)
    p.add_argument("--baseline", type=Path, default=BASELINE_PATH)
    p.add_argument("--update-baseline", action="store_true", help="write this run's metrics as the new baseline")
    p.add_argument("--no-gate", action="store_true", help="always exit 0 (report only)")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.provider:
        os.environ["LLM_PROVIDER"] = args.provider
    settings = load_settings()
    for provider in {settings.llm.provider, settings.judge.provider} & set(_KEY_ENV):
        if not os.getenv(_KEY_ENV[provider]):
            print(f"error: {_KEY_ENV[provider]} is not set (provider={provider})", file=sys.stderr)
            return 2

    items = sample(load_dataset(), args.sample, args.seed)
    cache = JudgeCache(args.cache_dir, enabled=not args.no_cache)
    judge_llm = make_llm(settings.judge, judge=True)
    judges = Judges(
        Judge(load_judge("correctness"), judge_llm, cache),
        Judge(load_judge("faithfulness"), judge_llm, cache),
    )
    started = time.perf_counter()
    records, metrics = run(
        items, build_pipeline(settings), judges, settings, set(load_corpus(CORPUS_DIR)), args.workers
    )

    meta = {
        "timestamp": datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC"),
        "git_sha": _git_sha(),
        "provider": settings.llm.provider,
        "model": settings.llm.model,
        "judge_model": settings.judge.model,
        "judges": {
            "correctness": judges.correctness.spec.fingerprint,
            "faithfulness": judges.faithfulness.spec.fingerprint,
        },
        "prompt_version": PROMPT_VERSION,
        "top_k": settings.top_k,
        "sample_size": len(items),
        "seed": args.seed,
        "dataset_sha": hashlib.sha256(DATASET_PATH.read_bytes()).hexdigest()[:12],
        "judge_cache_hits": cache.hits,
        "judge_cache_misses": cache.misses,
        "wall_time_s": round(time.perf_counter() - started, 1),
    }
    baseline = None if args.update_baseline else load_baseline(args.baseline)
    checks = gate(metrics, load_thresholds(args.thresholds), baseline)
    report = write_reports(records, metrics, checks, meta, args.out)
    if args.update_baseline:
        write_baseline(metrics, meta, args.baseline)
        print(f"baseline written to {args.baseline}")

    ok = passed(checks)
    print(summary_table(checks, metrics))
    print(f"\n{'PASSED' if ok else 'FAILED'} - report: {report}")
    return 0 if ok or args.no_gate else 1


if __name__ == "__main__":
    sys.exit(main())
