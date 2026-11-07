"""The full LLM evaluation as a pytest test. Deselected by default; run with:

LLM_PROVIDER=anthropic EVAL_SAMPLE=50 pytest -m eval
"""

import os

import pytest

from evals.run import main


@pytest.mark.eval
def test_llm_quality_gates():
    if os.getenv("LLM_PROVIDER", "fake") == "fake":
        pytest.skip("set LLM_PROVIDER=anthropic|openai to run the real evaluation")
    assert main([]) == 0, "quality gates failed; see evals/reports/report.md"
