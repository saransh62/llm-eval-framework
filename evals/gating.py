"""Threshold and baseline-regression gating."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import yaml

EVALS_DIR = Path(__file__).parent
THRESHOLDS_PATH = EVALS_DIR / "thresholds.yaml"
BASELINE_PATH = EVALS_DIR / "baseline.json"


@dataclass
class Check:
    metric: str
    value: float | None
    direction: str  # "min" (higher is better) or "max" (lower is better)
    threshold: float
    baseline: float | None
    margin: float
    status: str = "pass"  # pass | fail | skip
    reasons: list[str] = field(default_factory=list)


def load_thresholds(path: Path = THRESHOLDS_PATH) -> dict:
    data = yaml.safe_load(path.read_text())
    for name, rule in data["metrics"].items():
        if ("min" in rule) == ("max" in rule):
            raise ValueError(f"threshold for {name} needs exactly one of min/max")
    return data


def load_baseline(path: Path = BASELINE_PATH) -> dict | None:
    return json.loads(path.read_text()) if path.exists() else None


def gate(metrics: dict[str, float | None], thresholds: dict, baseline: dict | None) -> list[Check]:
    base_metrics = (baseline or {}).get("metrics", {})
    checks = []
    for name, rule in thresholds["metrics"].items():
        direction = "min" if "min" in rule else "max"
        check = Check(
            metric=name,
            value=metrics.get(name),
            direction=direction,
            threshold=float(rule[direction]),
            baseline=base_metrics.get(name),
            margin=float(rule.get("margin", thresholds.get("default_margin", 0.05))),
        )
        if check.value is None:
            check.status = "skip"
            check.reasons.append("not applicable to this sample")
            checks.append(check)
            continue
        better = (lambda a, b: a >= b) if direction == "min" else (lambda a, b: a <= b)
        if not better(check.value, check.threshold):
            check.reasons.append(f"{'below min' if direction == 'min' else 'above max'} {check.threshold:g}")
        if check.baseline is not None:
            allowed = check.baseline - check.margin if direction == "min" else check.baseline + check.margin
            if not better(check.value, allowed):
                check.reasons.append(f"regressed vs baseline {check.baseline:.4g} (margin {check.margin:g})")
        check.status = "fail" if check.reasons else "pass"
        checks.append(check)
    return checks


def passed(checks: list[Check]) -> bool:
    return all(c.status != "fail" for c in checks)


def checks_as_dicts(checks: list[Check]) -> list[dict]:
    return [asdict(c) for c in checks]


def write_baseline(metrics: dict, meta: dict, path: Path = BASELINE_PATH) -> None:
    path.write_text(json.dumps({"meta": meta, "metrics": metrics}, indent=2) + "\n")
