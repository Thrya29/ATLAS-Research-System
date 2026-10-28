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
