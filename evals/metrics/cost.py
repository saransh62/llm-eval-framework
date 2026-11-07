"""Metric 7 - latency and token cost."""

from __future__ import annotations


def cost_usd(input_tokens: int, output_tokens: int, price: dict[str, float]) -> float:
    """price is USD per million tokens: {"input": x, "output": y}."""
    return (input_tokens * price.get("input", 0.0) + output_tokens * price.get("output", 0.0)) / 1_000_000
