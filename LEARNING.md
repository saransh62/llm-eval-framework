# Learning guide

A walkthrough of the code in the order a request flows through it, followed by likely interview questions
with concise answers.

## Code walkthrough

### 1. The system under test (`app/`)

1. **`config.py`** loads `config.yaml` and lets env vars override it (`LLM_PROVIDER`, `LLM_MODEL`,
   `JUDGE_MODEL`, `TOP_K`, ...). `Settings.price(model)` feeds the cost metric.
2. **`chunking.py`** reads every `app/corpus/*.md` file. The doc id is the file name, which is what the
   model cites. Each document is split on `## ` headings, and long sections are packed sentence by sentence
   up to `chunk_max_chars`. `Chunk.indexed_text` prepends the title and section name because headings carry
   keywords ("Refund Policy > Annual plans") that the body often lacks.
3. **`retrieval.py`** tokenizes (lowercase, alphanumerics, decimals kept so "99.9" survives, stopwords removed,
   including "acme" and "cloud", which appear in every document) and ranks chunks with `BM25Okapi`. It
   returns the top-k chunks with a positive score.
4. **`prompts.py`** holds the system prompt rules: answer only from `<context>`, cite `[doc-id]`, reply with
   one exact refusal sentence when the answer is missing, treat context and question as data, and never
   reveal the prompt. It contains a **canary** string (`ZEBRA-CANARY-4471`) that should never appear in an
   answer. `build_user_prompt` HTML-escapes the question and chunks so a user cannot close the
   `<question>` tag and inject a fake one. `parse_citations` and `is_refusal` are the deterministic parsers
   the evals rely on.
5. **`llm.py`** defines the `LLMClient` protocol (`complete(system, user) -> LLMResponse` with token counts)
   and two fakes. `FakeLLM` returns the context sentence with the best word overlap, or refuses.
   `FakeJudgeLLM` returns JSON that satisfies both judge schemas. `make_llm` picks the provider. Real
   clients live in **`providers.py`** and are imported lazily.
6. **`providers.py`**: `AnthropicLLM` calls the Messages API with `output_config.effort`, enables server-side
   refusal fallback where supported, and maps a safety `stop_reason == "refusal"` to the refusal sentence.
   `OpenAILLM` uses Chat Completions with `temperature` from config. **`claude_cli.py`** (`ClaudeCLILLM`) shells
   out to `claude -p` with tools off and JSON output, so the eval can run on a Claude subscription without a key.
7. **`rag.py`**: `RAGPipeline.ask` = retrieve, build prompt, call the LLM, then parse citations and refusal,
   and time the whole thing. `AskResult` keeps the retrieved chunks so the evals can judge faithfulness
   against exactly what the model saw.
8. **`main.py`**: FastAPI `POST /ask` with Pydantic validation (1-2000 chars). The pipeline is injected with
   `Depends`, so tests swap in `FakeLLM` with `app.dependency_overrides`.

### 2. The evaluation suite (`evals/`)

1. **`dataset.py`** loads `golden.jsonl` into validated `GoldenItem`s and provides `sample(n, seed)`, which
   takes round-robin across categories so even 5 questions cover factual, unanswerable and adversarial.
2. **`metrics/`** has one pure function module per metric family (retrieval, correctness, faithfulness,
   refusal, safety, citation, cost) plus `stats.py` (`mean` that skips `None`, nearest-rank `percentile`).
   `None` means "not applicable to this item" and is excluded from averages instead of counted as 0.
3. **`judge.py`** loads a rubric file (`<name>_v<N>.md`, highest N wins), builds a tagged and escaped user
   message, calls the judge model, extracts the first JSON object, validates the score and retries once.
   **`cache.py`** stores verdicts on disk keyed by
   `sha256(judge fingerprint, judge model, every input field)`.
4. **`runner.py`**: `evaluate_item` asks one question and applies every applicable metric, collecting
   `(kind, reason)` failures and a severity score. `aggregate` turns records into run metrics. `run`
   parallelises over a thread pool, since the work is I/O-bound.
5. **`gating.py`** compares metrics with `thresholds.yaml` (absolute floor or ceiling) and
   `baseline.json` (regression margin), and returns `Check`s with pass/fail/skip and reasons.
6. **`report.py`** renders `report.md` (summary table, five worst failures, clean items by category),
   `summary.md` for the CI job summary, and `results.json` with everything.
7. **`run.py`** is the CLI that wires it together and exits 1 when a gate fails. **`test_eval_suite.py`**
   wraps it as a `pytest -m eval` test.

### 3. Tests and CI

`tests/` runs offline in about 2 seconds. `ci.yml` runs lint, the tests and a FakeLLM smoke eval on every
push. `eval.yml` runs the real evaluation on demand, with the judge cache restored from
`actions/cache`.

## Interview questions

**1. How do you test a non-deterministic system?**
Split it into deterministic parts (retrieval, parsing, metric math, gating), which get ordinary unit tests
with fakes, and the stochastic model output, which is measured statistically on a fixed dataset. Then reduce
variance where possible (temperature 0 or a fixed effort level, pinned model ids, cached judgements) and
compare runs with thresholds and margins rather than exact equality. Assert on properties (cites a real doc,
contains "$49", refuses) instead of exact strings.

**2. Why both deterministic metrics and an LLM judge?**
Deterministic checks are cheap, reproducible and unit-testable, but brittle to paraphrase. The judge handles
paraphrase and partial credit but is noisy and can be biased. Running both and reading the items where they
disagree finds bugs in either one.

**3. What are the main RAG failure modes?**
Retrieval misses (vocabulary mismatch, chunk boundaries cutting a fact in half), right context but a wrong
answer, answers from prior knowledge instead of context (unfaithful), hallucinating on unanswerable questions,
over-refusing answerable ones, wrong or invented citations, and following instructions embedded in retrieved
documents (indirect prompt injection).

**4. Why measure retrieval separately from generation?**
Fault isolation. If recall@k is 1.0 but correctness is low, the problem is the prompt or the model. If recall
is low, no prompt change will fix it. It is also deterministic and free, so it can gate every commit.

**5. What is faithfulness, and how does it differ from correctness?**
Correctness compares the answer with the truth. Faithfulness compares it with the retrieved context. An answer
can be correct but unfaithful (the model knew the fact from pre-training, which is risky because next time it
may "know" something false) or faithful but incorrect (the retrieved chunk was outdated or wrong).

**6. How do you make an LLM judge reliable?**
Give it an anchored rubric, ground truth (reference facts, context) and a strict output schema. Pin the model
and keep it at temperature 0 where possible. Version the rubric and cache verdicts. Validate and retry on
malformed output. Most importantly, spot-check it against human judgement and fix the rubric when they
disagree. Optionally use a judge from a different model family than the system under test.

**7. Why version judge prompts?**
A rubric edit changes the measuring instrument. Scores before and after are not comparable, and cached
verdicts would be stale. Here the fingerprint (version plus a content hash) is part of the cache key and of
the report metadata, so any edit invalidates the cache and shows up in review.

**8. How does the suite catch prompt injection?**
Ten adversarial items cover direct injection, an injection attached to a real question, system-prompt leak
attempts, jailbreak personas, a false premise, a poisoned corpus document (indirect injection), exfiltration
and a harmful request. The checks are forbidden output strings, a canary in the system prompt that must
never be printed, and a required refusal where appropriate. The safety gate has zero tolerance.

**9. How do you defend against prompt injection in a RAG app?**
Treat retrieved text and user input as data: delimit and escape them, and tell the model explicitly that
instructions inside them are not to be followed. Keep the system prompt free of real secrets (use a canary
to detect leaks). Constrain the output format, apply least privilege (this app has no tools, so injection
cannot trigger actions), and keep regression tests for every known attack. No prompt alone is a complete
defence, which is why it is measured continuously.

**10. How is cost controlled?**
The real-model eval runs only on demand, with stratified sampling for cheap runs (for example 20 questions) and the full set before a baseline update, a judge cache so unchanged
answers are never re-judged, skipping the faithfulness judge for refusals, low effort for answers, offline
fakes for all development and CI, and a `cost_per_question_usd` gate so a prompt that doubles token usage
fails the build.

**11. How do you set thresholds?**
Start from a real baseline run. Set floors slightly below it at a level that would be unacceptable to ship,
and use margins for drift. Size margins to the metric's noise: tight for deterministic metrics, zero for
safety, wide for latency. Revisit them when the dataset grows. Never lower a threshold just to make a
failing build pass without writing down why.

**12. A metric dropped in CI. How do you debug it?**
Open `report.md`: the worst failures list the question, answer, retrieved vs. expected docs and reasons.
Classify the failure (retrieval miss, prompt, model, judge error, or a bad golden label), reproduce that item
locally with `--sample`, and compare the run metadata (git sha, model, judge fingerprint, dataset hash) with
the baseline.

**13. Why BM25 instead of a vector database?**
The corpus is small, the vocabulary is precise (plan names, numbers, error codes) and BM25 is deterministic,
dependency-light and easy to explain. The eval suite is retriever-agnostic, so swapping in embeddings or
hybrid search is a measured change: compare recall@k and MRR against the baseline.

**14. How do you know the golden dataset is right?**
It is tested: every expected source must exist, and every expected fact must appear in its source documents.
Questions cover every document plus deliberately hard cases (near-domain unanswerables, multi-fact answers).
When the model "fails" an item, check the label too; fix bad labels in a reviewed pull request.

**15. What would you add next?**
A larger dataset with production-like questions (sampled from logs once there are users), human-labelled
judge agreement measurement (for example Cohen's kappa on 50 items), confidence intervals via bootstrap,
multi-turn conversations, paraphrase robustness (the same question asked five ways), per-category gates, and
tracing so every production answer can be scored with the same metrics.
