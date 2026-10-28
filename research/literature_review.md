# Literature review and research gap

This review positions ATLAS against prior work on tool selection, cost-aware routing, budget-aware agents and
non-stationary bandits. Bibliographic details were checked against arXiv, the ACL Anthology or PMLR on 2026-10-28.
Where a venue could not be verified, the entry says so.

## 1. Tool selection and tool retrieval

| Work | What it establishes | What it leaves open (for this project) |
|---|---|---|
| **MetaTool** — Huang et al., ICLR 2024 | A benchmark for *whether* an LLM should use a tool and *which* one, including similar-tool confusion and multi-tool scenarios. LLMs struggle with tool awareness and fine-grained selection. | Selection is scored as accuracy against a label. Runtime cost, latency, reliability and the *sufficiency* of a multi-tool plan are not part of the objective. |
| **ToolRet** — Shi et al., *Retrieval Models Aren't Tool-Savvy*, Findings of ACL 2025 ([2025.findings-acl.1258](https://aclanthology.org/2025.findings-acl.1258), [arXiv:2503.01763](https://arxiv.org/abs/2503.01763)) | 7.6k retrieval tasks over a 43k-tool corpus (plus >200k training instances). IR models that are strong on conventional benchmarks retrieve tools poorly. | Retrieval returns *relevant* tools. It does not decide whether they are authorised, healthy, compatible with each other, or worth their cost. |
| **Gorilla / BFCL** — Patil et al.; Berkeley Function-Calling Leaderboard | Correctness of function calls (serial, parallel, multi-turn, agentic); later versions report cost and latency. | Evaluates the *call*, given the tool set. It does not address choosing the tool set under runtime constraints. |
| **τ-bench** — Yao et al., 2024 | Tool–agent–user interaction with policy adherence and state-based grading; introduces pass^k reliability. | Reliability of the *agent*, not adaptive routing over tools whose own reliability changes over time. |
| **AgentBench** — Liu et al., ICLR 2024 | Broad multi-environment agent evaluation. | Not focused on tool routing efficiency. |
| **Toolformer** (Schick et al., 2023), **ToolLLM** (Qin et al., 2023) | Teaching models to call tools; large-scale API corpora with DFS-based solution search. | Optimise for solving the task, not for minimum sufficient computation. |

## 2. Cost-aware routing

| Work | What it establishes | Gap |
|---|---|---|
| **Query Routing for Homogeneous Tools** — Mu et al., Findings of EMNLP 2024 ([2024.findings-emnlp.598](https://aclanthology.org/2024.findings-emnlp.598), [arXiv:2406.12429](https://arxiv.org/abs/2406.12429)) | Predicts the performance *and* cost of homogeneous tools (e.g. alternative retrievers/search engines) and assigns queries cost-effectively. | Homogeneous alternatives for one capability. ATLAS handles *heterogeneous* tools with typed input/output dependencies, so the decision is over plans, not single tools. It also handles drift and hard constraints. |
| **FrugalGPT** (Chen et al., 2023), **RouteLLM** (Ong et al., 2024) | LLM cascades and learned routers that trade answer quality against model cost. | Route among *models*. The same economics apply to tools, but tools add dependencies, permissions and failure modes. |

## 3. Budget awareness and the value of more tool calls

| Work | What it establishes | Gap |
|---|---|---|
| **BATS — Budget-Aware Tool-Use Enables Effective Agent Scaling** — Liu et al., 2025 ([arXiv:2511.17006](https://arxiv.org/abs/2511.17006)) | Giving agents more tool calls does not by itself raise performance. A budget tracker plus budget-aware planning/verification improves the cost–performance trade-off for search agents. | Budget awareness inside the agent's reasoning loop. ATLAS makes the budget decision at the *router* level, with explicit, learned estimates of each additional call's value. |
| **When Do Tools and Planning Help LLMs Think?** — Ghoshal & Al-Bustami, 2026 ([arXiv:2601.02663](https://arxiv.org/abs/2601.02663)) | A cost- and latency-aware comparison of one-shot answering against a LangGraph plan–execute–replan agent with tools. More orchestration is not uniformly beneficial. | An empirical characterisation, not a routing policy. It motivates ATLAS's value-of-information stopping rule. |
| **ProtocolBench / ProtocolRouter** — Du et al., ICML 2026 ([arXiv:2510.17149](https://arxiv.org/abs/2510.17149), [PMLR v306](https://proceedings.mlr.press/v306/du26l.html)) | Benchmarks multi-agent communication protocols on success, latency, overhead and robustness. A constraint-aware router filters by hard requirements, then selects. | Selects *protocols*. Its "filter by hard constraints, then optimise" structure parallels ATLAS's feasibility filter plus constrained plan choice, applied here to tools. |
| *Tool and Agent Selection for LLM Agents in Production: A Survey* — Preprints.org, Dec 2025 (venue not verified) | Frames selection as a scalability (hundreds to thousands of tools) and reliability/authorisation problem. | A survey. It is useful for the production taxonomy (retrieval, authorisation, HITL). |

## 4. Bandits for non-stationary, combinatorial, constrained decisions

* **Discounted / sliding-window UCB** — Garivier & Moulines, 2011: regret guarantees under abruptly changing rewards via
  geometric discounting or a window.
* **Discounted Thompson sampling** — Raj & Kalyani, 2017: a Bayesian analogue. ATLAS discounts with *global* time, so unused
  arms regain variance and get re-explored.
* **Combinatorial (semi-)bandits** — Chen, Wang & Yuan, 2013 (CUCB); Kveton et al., 2015: per-base-arm feedback when a
  super-arm (here, a plan) is played.
* **Bandits with knapsacks** — Badanidiyuru, Kleinberg & Slivkins, 2013; **safe/conservative bandits** — Wu et al., 2016;
  Amani et al., 2019: resource and safety constraints during exploration.

## 5. The gap ATLAS targets

No surveyed work jointly addresses all of the following for **heterogeneous tools**:

1. **Plans, not tools.** Tools have typed inputs and outputs, so a sufficient answer can need a dependency-respecting set
   of tools (e.g. SQL → Python). The decision space is the set of feasible plans in an AND–OR graph.
2. **Hard constraints, not a scalarised reward.** Quality, reliability and latency-tail targets are constraints, and cost is
   minimised among plans that meet them. A cheap wrong answer can never "win" on a weighted sum.
3. **Non-stationarity.** Tool latency, failure rate and quality drift, so estimates must forget, and unused tools must be
   re-explored.
4. **Safe exploration.** Sensitive or unauthorised tools are never explored optimistically.
5. **Sufficiency.** After execution, decide whether *one more* call is worth its cost (value of information) instead of
   always calling more or never verifying.
6. **Learning from task-level feedback only.** Per-tool correctness is not observed, so credit has to be assigned.

ATLAS's contribution is an integrated, testable instantiation of (1)–(6). Section 6 of the results
(`benchmarks/reports/RESULTS.md`) measures how much each part contributes. **We do not claim** that any single
ingredient (Thompson sampling, discounting, branch-and-bound, VoI) is new. The claim is about the formulation and its
empirical behaviour in a controlled, reproducible environment.
