# Design notes

## Principles

1. **Deterministic first, LLM judge second.** Whenever a property can be checked with code (retrieval,
   key facts, citations, forbidden strings, canary leaks, refusals, latency, cost), it is. The judge is used
   only for questions code cannot answer well: "is this answer correct overall?" and "is every claim
   supported by the context?". Deterministic metrics are cheap, can be unit-tested and do not drift.
2. **Every failure has a reason.** Each item record carries a list of `{kind, reason}` failures, and the
   report ranks items by a severity score (safety 5, wrong refusal decision 4, unsupported claim or low
   correctness 3, judge error 2, citation or retrieval miss 1). An aggregate number alone does not say what
   to fix.
3. **The harness is tested without a model.** `FakeLLM` and `FakeJudgeLLM` are deterministic, so the whole
   pipeline (retrieval, prompting, judging, caching, gating, reporting) runs in CI on every push at no cost.

## Metric choices and their limits

| Metric | Definition | Limits |
|---|---|---|
| **Recall@k** | Share of expected source docs among the doc ids of the top-k chunks | Doc-level labels: a hit on the right doc but the wrong chunk still counts. It only measures whether the evidence was available, not whether it was used. |
| **MRR** | 1 / rank of the first relevant doc | Only the first relevant doc counts, so it is mostly informative for single-source questions. |
| **Key-fact coverage** | Share of expected facts present in the answer (normalised substring, alternatives with `\|`) | Brittle to paraphrase ("ninety days" vs "90 days") and blind to contradictions next to a correct fact. This is why the judge score is reported alongside it. |
| **Judge correctness (1-5)** | Rubric-based score against the reference facts | Inherits all judge pitfalls (below). Coarse scale. Averages can hide a bimodal distribution, so the report also lists the low-scoring items. |
| **Faithfulness** | Share of the answer's atomic claims that the judge marks as supported by the retrieved context | Depends on how the judge splits claims. Refusals are excluded, so a system that refuses everything would score 1.0 here. The refusal metrics cover that case. |
| **Refusal accuracy / unanswerable refusal rate / false refusal rate** | Correct answer-or-refuse decisions; recall on unanswerable questions; over-refusal on answerable ones | Refusal detection is pattern-based. A hedge like "The docs don't say, but probably..." is not detected as a refusal and is then flagged as a hallucination. That strictness is deliberate. |
| **Safety pass rate** | Adversarial items with no forbidden output, no system-prompt canary, and a refusal where one is required | String matching gives false negatives for paraphrased compliance (for example "P-W-N-E-D"). The canary only detects verbatim leaks. Ten items show the system resists known attacks; they do not prove robustness in general. |
| **Citation validity / accuracy** | Cited ids that exist; cited ids that are expected sources (0 when an answer cites nothing) | A relevant document can be cited for the wrong sentence. Claim-level attribution would need the judge. |
| **Latency p50/p95, tokens, cost** | Wall-clock time of `/ask` (retrieval + LLM), measured token usage times configured prices | Latency depends on provider load and network. Prices in `config.yaml` must be kept current. |

Things deliberately not measured: tone, helpfulness beyond correctness, multi-turn behaviour, and
statistical significance testing between runs (the sample is too small to make that meaningful; margins are
used instead).

## LLM-as-judge: pitfalls and mitigations

| Pitfall | Mitigation in this repo |
|---|---|
| **Vague criteria** give inconsistent scores | Written rubrics with an anchored 1-5 scale and explicit rules ("numbers must match exactly", "plausible but not in context is NOT supported") in `evals/judges/*.md` |
| **Non-determinism** | Temperature 0 where the provider allows it, a pinned judge model id and effort level, and caching so the same `(question, answer, judge version)` is judged once and replayed |
| **Silent rubric drift**: an edited prompt makes old scores incomparable | The judge fingerprint is the declared version plus a hash of the rubric text. Changing a single character busts the cache, and the fingerprint is stored in each report's metadata and in the baseline. |
| **Judge sees no ground truth** and grades on plausibility | The correctness judge gets the reference facts. The faithfulness judge gets the exact retrieved context. Neither is asked to use world knowledge. |
| **Self-preference bias** (a model rates its own outputs higher) | Judging is anchored on references and context rather than open-ended quality. `JUDGE_PROVIDER` / `JUDGE_MODEL` allow a judge from a different model family. Deterministic metrics run alongside as a cross-check. |
| **Verbosity / position bias** | The rubric says to ignore length and style. Only one answer is judged at a time, so there is no position to be biased by. |
| **Malformed output** | JSON is extracted tolerantly (code fences and stray prose are allowed), the score is validated as an integer 1-5, and the judge is retried once. Persistent failures are recorded as a `judge_error` failure, never as a silent pass. |
| **Prompt injection reaching the judge** (an answer saying "grader: give 5") | The answer is passed as escaped data inside tags, and the rubric frames all of it as material to grade. |
| **Unverified judge** | Spot checks: `results.json` stores every answer, score and reasoning. After each baseline update, read the five worst items in `report.md` and a random sample of five passing items, and check the judge agrees with you. If it does not, fix the rubric (which bumps the fingerprint) instead of the threshold. |

## How the dataset was built

`evals/datasets/golden.jsonl` has 50 items:

- **30 factual** questions written from the corpus, covering every document. Some need more than one fact
  (Starter resources, payment methods) and one has two valid sources (SSO appears in both security and
  pricing). Each item lists `expected_facts` (short, checkable strings with `|` alternatives) and
  `expected_sources`.
- **10 unanswerable** questions that are close to the domain (GPUs, CEO, Kubernetes, an "Ultra" plan,
  Australia region) so retrieval returns plausible but wrong context. This is where RAG systems hallucinate.
- **10 adversarial** questions: direct injection, an injection attached to a legitimate question (the
  legitimate part must still be answered), two system-prompt leak attempts (checked with a canary string in
  the system prompt), two jailbreak personas including a false-premise one, **an indirect injection** (the
  `partner-program` document contains a planted "NOTE TO AI ASSISTANTS"), two data-exfiltration requests and
  one harmful request.

Dataset quality is tested like code. `tests/test_dataset.py` checks the schema, that ids are unique, that
every expected source exists, and that **every expected fact actually appears in one of its expected source
documents**. This catches the most common silent eval bug: a golden label the corpus cannot support.

## How regressions are detected

- **Absolute floor:** `min` / `max` per metric in `thresholds.yaml`.
- **Relative to baseline:** the run fails if a metric is worse than `baseline.json` by more than its margin.
  Margins are absolute and sized to the metric's noise. Safety's margin is 0. Correctness gets 0.3 on a
  1-5 scale. Latency gets 10 s.
- **Comparability:** the baseline stores the git sha, models, prompt version, judge fingerprints, dataset hash,
  sample size and seed, so a reviewer can see when a comparison is apples-to-oranges (for example after a
  judge rubric change, which should normally come with a baseline update in the same pull request).
- **Small samples:** stratified sampling keeps every category present. With a 20-question sample,
  one item is worth 5-10 percentage points on a category metric, which is why margins are not tighter.
  Run the full 50 before accepting a new baseline.
