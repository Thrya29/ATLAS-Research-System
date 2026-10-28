# ATLAS — Adaptive Tool Learning and Allocation System

**Research question.** Can an agent choose the smallest, cheapest, most reliable *set* of tools that is sufficient
to answer a query, and keep making good choices as tool latency, failure rates and quality drift?

ATLAS is a tool router for LLM agents (LangChain/LangGraph integration included). For each query it picks the
**cheapest plan that meets explicit quality, reliability and latency-tail constraints**. It learns tool behaviour
online and decides after execution whether **one more call is worth its cost**.

* **Algorithm:** AND–OR plan search over typed tool capabilities, combined with constrained, time-decayed Thompson
  sampling, fallback replanning and a value-of-information stopping rule. → [research/algorithm_design.md](research/algorithm_design.md)
* **Evidence:** 6 baselines + a privileged oracle, 9 ablations, drift, scaling and SLO sweeps. Results use 20 seeds,
  common random numbers, bootstrap CIs and Holm-corrected paired tests. → [benchmarks/reports/RESULTS.md](benchmarks/reports/RESULTS.md)
* **Production:** a FastAPI service with RBAC, audit log, Prometheus metrics, a feedback loop, state snapshots
  (file or Redis), hardened SQL and Python tools, Docker Compose and CI. → [docs/](docs/)

## Results (20 seeds, held-out query phrasings)

Mean [95% bootstrap CI over seeds]. Cost is in milli-USD per *successful* task.

| Router | E1 stationary: success | E1: cost / success | E2 drift: success | E2: cost / success |
|---|---|---|---|---|
| B0 Single tool per category | 0.354 [0.349, 0.359] | 6.34 | 0.348 [0.344, 0.352] | 6.41 |
| B1 Rules (keyword, permission-aware) | 0.486 [0.480, 0.492] | 6.12 | 0.475 [0.471, 0.479] | 6.26 |
| B2 Model-based (supervised on oracle plans) | 0.833 [0.830, 0.837] | 3.97 | 0.724 [0.720, 0.729] | 4.54 |
| B3 Retrieval (BM25 + vector, top-2) | 0.296 [0.292, 0.300] | 11.57 | 0.293 [0.289, 0.297] | 11.69 |
| B4 Static cost-aware (same planner, frozen stats) | 0.864 [0.860, 0.869] | 4.88 | 0.822 [0.816, 0.828] | 4.95 |
| B5 Contextual bandit (scalarised reward) | 0.839 [0.836, 0.842] | 4.23 | 0.803 [0.800, 0.806] | 4.54 |
| **ATLAS** | **0.899 [0.897, 0.902]** | **3.70** | **0.873 [0.871, 0.875]** | **4.23** |
| Oracle (true requirements + true current params) | 0.894 [0.892, 0.897] | 3.28 | 0.891 [0.889, 0.893] | 3.71 |

Every ATLAS-vs-baseline difference in success and cost per success above is significant after Holm correction
(p < 0.001, n = 20 paired seeds).

![Rolling success under drift](benchmarks/reports/figures/e2_drift_success.png)

### Hypotheses (fixed before the reported runs; see [research/hypotheses.md](research/hypotheses.md))

| | Hypothesis | Outcome |
|---|---|---|
| H1 | Higher success than the rules router and the model-based router | **Supported.** +0.41 vs B1 and +0.07 vs B2 (stationary); +0.40 and +0.15 under drift |
| H2 | Lower cost per successful task than both | **Supported.** −2.42 / −0.27 m$ (stationary); −2.03 / −0.31 m$ (drift) |
| H3 | The gain is more than "using history": beats the static cost-aware router under drift | **Supported.** +0.051 success, −0.72 m$ cost / success vs B4 |
| H4 | Beats a contextual bandit without plan structure or constraints | **Supported.** +0.070 success, −0.31 m$ vs B5 |
| H5 | Plan-graph search and VoI stopping each contribute | **Supported.** Removing the plan graph costs −0.268 success. Removing VoI costs −0.055 success, with no significant change in cost / success |
| H6 | Routing overhead stays at a few ms up to 500 tools | **Not supported as stated.** p95 grows from 4.6 ms (12 tools) to 27 ms (512 tools), and success falls from 0.90 to 0.78 as hundreds of near-duplicate tools must be explored (B5 falls from 0.82 to 0.49) |

### What the results do *not* show (read before quoting numbers)

* **More calls, not fewer.** ATLAS makes ~2.0 calls per task vs ~1.6 for B4/B5. It has 0.47–0.52 *strict*
  unnecessary calls per task vs ~0.12, because verification calls count as unnecessary by design. That is the price
  of the VoI rule, and it buys +5.5 points of success at an unchanged cost per success. [E5](benchmarks/reports/RESULTS.md#e5--slo--value-of-success-sweep-under-drift)
  shows the knob: lowering `min_quality` to 0.6 gives 0.82 success at 3.46 m$ per success. That still beats
  B4 and B5 under drift on both axes.
* **Tail latency.** Under drift, ATLAS's p95 latency (1432 ms) is worse than B5's (1256 ms), and its
  P(latency > budget) is 4.4% vs 3.1%, because verification adds a serial step.
* **Recovery speed.** Tasks-to-recover after the drift onset (384) is not better than B4/B5 (308–361). ATLAS's
  advantage comes from staying higher throughout, not from recovering faster.
* **Component effects.** Several components have no measurable effect in this environment: learned reliability (A2),
  learned latency (A6), the feasibility filter (A7, ≤0.5 points), and fallback replanning (A8, 0.4 points).
  **Per-family context (A1) slightly *hurts*** (removing it gives +0.7 points). Family predictions are imperfect, and
  splitting the data slows adaptation.
* **Simulation.** Tool parameters are hand-specified (documented in [docs/experiments.md](docs/experiments.md)), and
  task understanding is a small text classifier (0.88 exact-set accuracy on held-out phrasings). These numbers show
  behaviour in a controlled environment. They are not production measurements.

## Quick start

```bash
python -m venv .venv && . .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -e ".[dev,langchain]"
pytest -q                                              # 61 tests: unit, property, integration, API, LangGraph

uvicorn atlas.api.app:app --port 8000                  # demo service (key: dev-key)
curl -s localhost:8000/v1/execute -H 'X-API-Key: dev-key' -H 'Content-Type: application/json' \
     -d '{"query": "Calculate the percentage change in revenue between Q1 and Q2"}'

docker compose up --build                              # API + Redis + Prometheus
```

## Reproducing every number

```bash
python -m benchmarks.experiments.run_all --workers 5   # E1–E5, seeds 1–20 (~26 min on 6 cores); resumable
python -m benchmarks.experiments.report                # writes benchmarks/reports/RESULTS.md + figures
```

Seed 0 was used for development and is excluded. The simulator uses common random numbers, so every router sees
identical tasks and identical per-(task, tool, attempt) outcomes.

## Using ATLAS in a LangChain agent

```python
from atlas.integrations.langchain import AtlasToolSelector, atlas_node

selector = AtlasToolSelector([(my_tool, ToolSpec(name="my_tool", description="...",
                                                 provides=frozenset({"web_facts"}), cost=0.004))], understanding)
tools, decision = await selector.select(query)        # bind only these to the LLM: llm.bind_tools(tools)
await selector.feedback(decision.context.request_id, success=True)
graph.add_node("atlas", atlas_node(selector))         # or let ATLAS execute the plan inside LangGraph
```

## Repository map

| Path | Contents |
|---|---|
| [research/](research/) | Literature review and gap, formal problem, algorithm design, pre-registered hypotheses |
| [atlas/](atlas/) | Router, planner, estimators, executor, tools, API, persistence, LangChain integration |
| [benchmarks/](benchmarks/) | Simulator and task families, baselines, harness, statistics, experiments, report |
| [tests/](tests/) | Unit, property-based (plan search vs brute force), integration, API and security tests |
| [docs/](docs/) | Architecture, API, deployment, experimental methodology |
| [deployment/](deployment/) | Dockerfile, Prometheus config |

## License

MIT
