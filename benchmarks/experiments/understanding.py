"""E6: does ATLAS's advantage depend on the task-understanding stage?

    python -m benchmarks.experiments.understanding --prepare   # run the local LLM once (needs ATLAS_LLM_MODEL)
    python -m benchmarks.experiments.understanding             # route with every understanding variant

Understanding variants, all evaluated on the same held-out test queries:

* ``classifier``: the TF-IDF label-powerset model used everywhere else.
* ``llm``: Qwen2.5-0.5B-Instruct (Q4_K_M GGUF) on CPU via llama.cpp, few-shot,
  with every *training* phrasing of each family as an example. Its outputs are cached in
  ``reports/llm_understanding_cache.json`` so the routing runs are reproducible
  without the model.
* ``oracle``: the true family.
* ``noisy-p``: the true family, replaced by a random other family with
  probability p (a controlled accuracy sweep).
"""

from __future__ import annotations

import argparse
import json
import os
import random
import time
import zlib
from pathlib import Path

import numpy as np

from atlas.understanding.classifier import Understanding
from atlas.understanding.llm import LLMUnderstanding
from benchmarks.harness import BASE_CTX, run, train_understanding
from benchmarks.metrics import aggregate
from benchmarks.simulator.tasks import FAMILIES, TEMPLATES, _fill, generate
from benchmarks.stats import bootstrap_ci

OUT = Path(__file__).resolve().parents[1] / "reports"
CACHE = OUT / "llm_understanding_cache.json"
SEEDS, N = list(range(1, 6)), 400
ROUTERS = ["ATLAS", "B5", "B4", "B0"]
DESCRIPTIONS = {
    "revenue_growth": "growth or percentage change of one of our business metrics over time.",
    "cohort_analysis": "statistical analysis (cohorts, retention, trends, regressions) of our customer or metric data.",
    "kpi_today": "the current, live value of one of our metrics today.",
    "policy_lookup": "what an internal company policy or handbook says.",
    "incident_triage": "diagnosing errors or an outage in one of our services from logs and runbooks.",
    "market_compare": "comparing our metrics with another company's public or market figures.",
    "stock_quote": "a company's current share price or stock quote.",
    "breaking_news": "latest news or headlines about an external topic.",
    "claim_check": "fact-checking a public statistic or claim with numbers.",
    "general_knowledge": "general knowledge that needs no company data or live information.",
    "metric_explainer": "explaining why one of our metrics changed, using data and internal documents.",
}


def test_tasks():
    return {s: generate(N, s, "test") for s in SEEDS}


def llm(model_path: str | None = None) -> LLMUnderstanding:
    rng = random.Random("llm-examples")
    examples = {f: [_fill(t, rng) for t in TEMPLATES[f]["train"]] for f in FAMILIES}
    return LLMUnderstanding(model_path or "", FAMILIES, DESCRIPTIONS, examples, cache_path=CACHE)


class OracleUnderstanding:
    """True family for known test queries; with probability p, a random wrong family instead."""

    def __init__(self, tasks, p: float = 0.0) -> None:
        self.family = {t.query: t.family for ts in tasks.values() for t in ts}
        self.p = p

    def __call__(self, query: str) -> Understanding:
        fam = self.family[query]
        r = random.Random(zlib.crc32(query.encode()))
        if r.random() < self.p:
            fam = r.choice([f for f in FAMILIES if f != fam])
        return Understanding(FAMILIES[fam], {}, fam, {fam: 1.0})


def prepare(model_path: str) -> None:
    m = llm(model_path)
    queries = sorted({t.query for ts in test_tasks().values() for t in ts})
    todo = [q for q in queries if q not in m.cache]
    print(f"{len(queries)} unique queries, {len(todo)} to classify", flush=True)
    for i, q in enumerate(todo, 1):
        m.family(q)
        if i % 50 == 0:
            m.save_cache()
            print(f"{i}/{len(todo)} median {np.median(m.latencies_ms):.0f} ms", flush=True)
    m.save_cache()
    if m.latencies_ms:
        lat = {
            "median_ms": float(np.median(m.latencies_ms)),
            "p95_ms": float(np.percentile(m.latencies_ms, 95)),
            "n": len(m.latencies_ms),
        }
        (OUT / "llm_understanding_latency.json").write_text(json.dumps(lat, indent=1))


def accuracy(u, tasks) -> dict[str, float]:
    ts = [t for v in tasks.values() for t in v]
    t0 = time.perf_counter()
    preds = [u(t.query) for t in ts]
    ms = (time.perf_counter() - t0) * 1000 / len(ts)
    return {
        "family_acc": float(np.mean([p.family == t.family for p, t in zip(preds, ts, strict=True)])),
        "exact_set_acc": float(np.mean([p.types == t.required for p, t in zip(preds, ts, strict=True)])),
        "ms_per_query": ms,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prepare", action="store_true")
    args = ap.parse_args()
    if args.prepare:
        prepare(os.environ["ATLAS_LLM_MODEL"])
        return
    tasks = test_tasks()
    variants = {
        "classifier": train_understanding(),
        "llm": llm(),
        "oracle": OracleUnderstanding(tasks),
        "noisy-0.2": OracleUnderstanding(tasks, 0.2),
        "noisy-0.4": OracleUnderstanding(tasks, 0.4),
    }
    out: dict = {"accuracy": {k: accuracy(u, tasks) for k, u in variants.items()}, "routing": {}}
    lat = OUT / "llm_understanding_latency.json"
    if lat.exists():
        out["llm_latency"] = json.loads(lat.read_text())
    for name, u in variants.items():
        for r in ROUTERS:
            per_seed = [aggregate(run(r, s, N, understanding=u), BASE_CTX.latency_budget_ms) for s in SEEDS]
            out["routing"][f"{name}|{r}"] = per_seed
            print(name, r, f"success {np.mean([a['success'] for a in per_seed]):.3f}", flush=True)
    (OUT / "E6.json").write_text(json.dumps(out, indent=1, default=float))
    lines = [
        "## E6 — Sensitivity to the understanding stage",
        "",
        f"Stationary, {len(SEEDS)} seeds × {N} held-out-phrasing tasks. Mean [95% bootstrap CI].",
        "",
        "| Understanding | Family acc. | Exact-set acc. | ms / query | " + " | ".join(f"{r} success" for r in ROUTERS) + " |",
        "|---|---|---|---|" + "---|" * len(ROUTERS),
    ]
    for name in variants:
        a = out["accuracy"][name]
        ms = out.get("llm_latency", {}).get("median_ms", a["ms_per_query"]) if name == "llm" else a["ms_per_query"]
        cells = []
        for r in ROUTERS:
            m, lo, hi = bootstrap_ci([x["success"] for x in out["routing"][f"{name}|{r}"]])
            cells.append(f"{m:.3f} [{lo:.3f}, {hi:.3f}]")
        lines.append(f"| {name} | {a['family_acc']:.3f} | {a['exact_set_acc']:.3f} | {ms:.1f} | " + " | ".join(cells) + " |")
    (OUT / "E6_understanding.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("wrote", OUT / "E6_understanding.md")


if __name__ == "__main__":
    main()
