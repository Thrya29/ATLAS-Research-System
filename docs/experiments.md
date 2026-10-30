# Experimental methodology

## Environment

`benchmarks/simulator` defines 12 tools with overlapping capabilities. Each has a ground-truth latency distribution
(log-normal), failure rate (half errors, half hangs that hit the timeout) and per-family answer quality. Highlights:

* `sql_primary` vs `sql_replica`: same capabilities. The replica is 4× cheaper but stale for real-time KPI questions
  (quality 0.55 on `kpi_today`).
* `python_sandbox` vs `calculator`: the calculator is ~30× cheaper and near-instant, but only correct for simple
  arithmetic (0.25 quality on cohort analysis). It needs `scalars` while Python needs a `table` (typed dependencies).
* Three web-search providers with different cost/latency/quality profiles. Two also provide market data at lower
  quality than the licensed `fin_api`, which only `finance`/`admin` roles may call (35% of requests).
* `log_search` vs the slow, cheap `log_archive`; `vector_docs` vs the low-quality `vector_legacy` index.

11 task families cover 0–3 required capability types, including a no-tool family. Training (classifier, supervised
router, rules) uses 4 phrasing templates per family. **Evaluation uses 2 disjoint held-out templates per family.**

### Drift schedule (E2, E3, E5; 3000 tasks)

| Tasks | Change |
|---|---|
| 1000–2000 | `search_b` overloaded: failure 3% → 50%, latency ×3, then it recovers |
| 1000–3000 | `sql_replica` replication lag: latency ×4, quality ×0.6 |
| 1800–2100 | `vector_docs` outage: failure 95% |
| 2000–3000 | `python_sandbox` latency ×2.5 (noisy neighbour) |
| 2200–3000 | `search_a` becomes 2× faster (upgrade) |

## Fairness controls

* **Common random numbers.** Latency, failure and correctness are drawn from generators keyed by
  (seed, task, tool, attempt), so routers that make the same choice see the same outcome.
* **Same inputs for every router.** All routers share the executor (1 retry, backoff, circuit breakers), the task
  stream and the 300-task offline warm-up log (collected by a uniform-random feasible-plan policy on stationary tools).
  Learned routers (B4, B5, ATLAS) start from that same log.
* **Same understanding model.** B0, B5 and ATLAS share the same task-understanding model. B2 uses the same model class
  but is trained end-to-end to predict the **oracle's** tool set, which gives it privileged supervision.
* **Rules not tuned on test data.** B1's rules were written from the training templates and are permission-aware.
* **Development seed excluded.** All tuning used seed 0. Reported runs use seeds 1–20.

## Baselines

| ID | Router | What it represents |
|---|---|---|
| B0 | Single tool per predicted family | Minimal static routing |
| B1 | Keyword rules | "Simple rules-based router" |
| B2 | Supervised set classifier (query + role → tool set), trained on oracle plans | "Reasonable model-based baseline" (offline-optimal imitation) |
| B3 | Hybrid BM25 + vector retrieval, top-2 tools | Embedding-retrieval routing (cf. ToolRet) |
| B4 | ATLAS planner with frozen historical averages; no exploration, VoI or replanning | Isolates the value of *adaptation* |
| B5 | Discounted Thompson sampling per (family, tool) on a scalarised reward | Standard contextual bandit without plan structure or constraints |
| Oracle | True requirements and true *current* parameters; min-cost feasible plan; no verification | Privileged reference, not an upper bound on success (it never verifies) |

## Statistics

Each run yields one value per seed and metric. Reported cells are the mean with a 95% percentile-bootstrap CI over
seeds (10,000 resamples). Comparisons use the paired difference per seed, with a bootstrap CI, a Wilcoxon signed-rank
p-value with Holm correction inside each table, and $d_z$ = mean difference / SD of differences.

## Reproduce

```bash
pip install -e ".[dev]"
python -m benchmarks.experiments.run_all --workers 5     # all experiments, seeds 1-20 (~26 min on 6 cores; resumable)
python -m benchmarks.experiments.report                  # RESULTS.md + figures
python -m benchmarks.experiments.run_all --quick         # 3 seeds, ~13 min
```

## Live evaluation (real tool execution)

`benchmarks/live` routes the same kinds of decisions over **tools that really execute**, on a generated company
world (`benchmarks/live/world.py`).

**The tools:**

| Capability | Tools |
|---|---|
| `scalars` / `table` | `sql_primary` (SQLite primary) and `sql_replica`, a stale copy whose latest-quarter figures are preliminary |
| `internal_docs` | `docs_hybrid` (BM25 + character n-gram fusion over current policies) and `docs_archive` (cheaper BM25 over an archive that also holds superseded versions with identical wording) |
| `logs` | `log_recent` (BM25 over the last two days) and `log_scan` (exact scan of the 30-day archive) |
| `numeric_result` | `calculator` (computes percentage changes only) and `python_sandbox` (isolated subprocess) |

**What is real:** the tools' outputs, errors and execution time. Answers are graded against the generator's ground
truth, which no tool computes.

**What is injected:** network latency in front of remote tools, transient failures, and a drift schedule. The schedule
has three events: a primary-database incident (30–65% of the stream, 60% failures), a noisy neighbour on the sandbox
(+900 ms from 50%), and a degraded search cluster (×12 latency from 65%). Injection is seeded per (task, tool,
attempt), so every router faces identical faults.

**The routers:**

* ATLAS, and ATLAS v1 (without A10).
* B4 and B5.
* Two keyword-rule routers: one always uses the best-quality tools, one always uses the cheapest.
* Every router is warm-started from the same 150-task uniform-policy log.

**Protocol:** 10 seeds × 400 held-out-phrasing tasks. The understanding model is trained on separate training
phrasings.

```bash
python -m benchmarks.live.run --seeds 1-10 --tasks 400      # ATLAS_LIVE_DIR sets where the world is written
```

## E6 — Sensitivity to the understanding stage

The same simulator and routers (stationary, seeds 1–5, 400 tasks), with the understanding stage replaced by one of:

* the TF-IDF classifier;
* a local LLM: Qwen2.5-0.5B-Instruct Q4_K_M via llama.cpp on CPU. It uses grammar-constrained decoding over family
  names, with the training phrasings as examples;
* the true family (oracle);
* the true family corrupted with probability 0.2 or 0.4.

The LLM's answers are cached in `reports/llm_understanding_cache.json`, so the routing runs reproduce without the model.

```bash
pip install -e ".[llm]"
ATLAS_LLM_MODEL=/path/to/qwen2.5-0.5b-instruct-q4_k_m.gguf python -m benchmarks.experiments.understanding --prepare
python -m benchmarks.experiments.understanding
```
