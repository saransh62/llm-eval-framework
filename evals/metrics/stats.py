"""Small statistics helpers shared by the metric modules."""

from __future__ import annotations

import math
from collections.abc import Iterable


def mean(values: Iterable[float | None]) -> float | None:
    """Mean of the non-None values, or None when there are none (metric not applicable to this sample)."""
    vals = [v for v in values if v is not None]
    return sum(vals) / len(vals) if vals else None


def percentile(values: Iterable[float], pct: float) -> float | None:
    """Nearest-rank percentile (pct in 0..100)."""
    vals = sorted(values)
    if not vals:
        return None
    rank = max(1, math.ceil(pct / 100 * len(vals)))
    return vals[rank - 1]
