"""Metric 3 - faithfulness / groundedness: share of answer claims the judge found supported by the context."""

from __future__ import annotations


def supported_fraction(claims: list[dict]) -> float | None:
    """claims: [{"claim": str, "supported": bool}, ...] as returned by the faithfulness judge."""
    if not claims:
        return None
    return sum(1 for c in claims if c.get("supported") is True) / len(claims)


def unsupported_claims(claims: list[dict]) -> list[str]:
    return [str(c.get("claim", "")) for c in claims if c.get("supported") is not True]
