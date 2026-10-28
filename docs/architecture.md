# Architecture

```
            ┌──────────────┐  X-API-Key → role (RBAC)          ┌──────────────────────┐
 client ───▶│ FastAPI      │──────────────────────────────────▶│ audit log (JSON)     │
            │ atlas/api    │  request id, Prometheus metrics    └──────────────────────┘
            └──────┬───────┘
                   ▼
   ┌────────────────────────────────────────── ATLASRouter (atlas/router.py) ───────────────────────────┐
   │ 1 TaskUnderstanding        query → required types Ŷ, family f   (atlas/understanding)              │
   │ 2 Candidate filter         capability index ∩ RBAC ∩ circuit health ∩ input satisfiability          │
   │ 3 Posterior draw           decayed Beta (reliability), hierarchical Beta (quality per family),       │
   │                            decayed log-normal (latency)                     (atlas/bandits)          │
   │ 4 Plan search              AND-OR branch-and-bound, k cheapest feasible tool sets  (atlas/planning)  │
   │ 5 Constrained selection    min E[cost] s.t. Q ≥ Qmin, R ≥ 1-ε, P(L>Lmax) ≤ δ  (Monte-Carlo DAG)     │
   └───────────────┬──────────────────────────────────────────────────────────────────────────────────┘
                   ▼ Plan (DAG)
   ┌────────────────────────── Executor (atlas/execution) ────────────────────────────┐
   │ topological levels run concurrently (asyncio) · per-tool timeout · retries with   │
   │ exponential backoff + full jitter · circuit breakers · RBAC re-check ·            │
   │ fallback hook (replan failed step) · follow-up hook (VoI verification)            │
   └───────────────┬──────────────────────────────────────────────────────────────────┘
                   ▼
   Tool adapters (atlas/tools): read-only SQLite · BM25 docs/web/log search · calculator · Python sandbox
   LangChain tools via atlas/integrations/langchain.py
                   ▼
   ExecutionReport → router.observe() (reliability, latency) ; /v1/feedback → quality + calibration
   State snapshots → FileStore | RedisStore (atlas/persistence.py)
```

## Module map

| Path | Responsibility |
|---|---|
| `atlas/core` | Data model (`ToolSpec`, `Plan`, `Decision`, `ExecutionReport`) and the registry with its inverted capability index |
| `atlas/understanding` | Query → capability types (label-powerset logistic regression) and task family |
| `atlas/retrieval` | BM25 + char-n-gram hybrid retrieval with RRF (the B3 baseline and the corpus tools) |
| `atlas/planning` | AND-OR plan enumeration (branch-and-bound) and DAG construction |
| `atlas/bandits` | Time-decayed Bayesian estimators, hierarchical quality model |
| `atlas/router.py` | The ATLAS decision procedure, fallback and VoI hooks, learning and credit assignment |
| `atlas/execution` | Async DAG executor, retry policy, circuit breakers |
| `atlas/tools` | Production tool adapters and their security controls |
| `atlas/service.py`, `atlas/api` | Service wiring and the HTTP API |
| `atlas/persistence.py` | Learned-state snapshots (JSON) to a file or Redis |
| `atlas/integrations` | LangChain tool adapter, tool narrowing, LangGraph node |
| `benchmarks/` | Simulator, task families, baselines, harness, statistics, experiments, report |

## Relationship to adaptive parallel scheduling

ATLAS decides **what** to execute (which tools, in which dependency structure, and whether to add a verification
call). The executor here uses simple level-by-level parallelism. A critical-path-aware scheduler that decides **how**
to run a DAG efficiently can sit underneath: it would consume `Plan.steps` / `depends_on` unchanged.

## Concurrency and state

Routing is a few milliseconds of CPU and mutates the posteriors, so the service serialises `route` / `observe` behind
an `asyncio.Lock`. Tool execution happens outside the lock. The deployed configuration runs a single API worker per
replica. Running several replicas means each learns locally and snapshots to Redis on shutdown (last writer wins).
Merging sufficient statistics across replicas is additive, so a periodic merge is a straightforward extension, but it
is **not implemented**.
