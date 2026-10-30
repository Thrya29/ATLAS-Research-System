# ATLAS algorithm design

Code: [`atlas/router.py`](../atlas/router.py), [`atlas/planning/graph.py`](../atlas/planning/graph.py),
[`atlas/bandits/estimators.py`](../atlas/bandits/estimators.py), [`atlas/execution/executor.py`](../atlas/execution/executor.py).

## Pseudocode

```
ROUTE(q, x):
  τ ← τ + 1
  Ŷ, f ← UNDERSTAND(q)                          # label-powerset classifier + family classifier
  if Ŷ = ∅: return EMPTY_PLAN                   # answer directly, no tool
  C ← {}                                         # candidates per type
  for t in REGISTRY:
      if role ∉ ρ_t or circuit(t)=open or some req(t) has no provider: reject(t, reason); continue
      for y in prov(t): C[y].append(t)
  for y: keep top-m of C[y] by posterior-mean utility r̄·θ̄·V − c̄
  for each candidate t:                          # one Thompson draw per decision
      explore ← ¬sensitive(t)
      r̃_t ~ Beta_decayed(rel_t)     or mean if ¬explore
      θ̃_t ~ Beta_decayed(θ_{t,f} | prior = θ_t)   (hierarchical pooling)
      μ̃_t ~ N(μ̂_t, σ̂_t²/(n_0+n_t))
  Plans ← BRANCH_AND_BOUND(Ŷ, C, cost = c_t·E[attempts | r̃_t], k)
  for P in Plans: score R(P), Q(P), C(P), Pr[L>L_max] by Monte Carlo over the DAG
  return argmin_{P feasible} C(P)  else  argmax_P R·Q·(1−tail)

EXECUTE(P):  level-by-level async; per step: RBAC check → circuit check → attempt with timeout → retry (backoff+jitter)
             on terminal failure: FALLBACK = best remaining provider of the step's types (replanning)
             after all levels: FOLLOWUP = VoI rule (see problem_formulation.md)

LEARN(decision, report, success):
  for each call: rel_t ← update(ok); if ok or timeout: lat_t ← update(duration)
  if all Ŷ supplied and success observed:
      w ← responsibilities ∝ (1 − confidence) over the evidence chain   # soft credit assignment
      θ_{t,f} ← update(1) for all contributors if success, else update(0, weight w_t)
      for each answer superseded by a more confident one (never graded):          # censoring correction
          θ_{t,f} ← update(calibration[bin(c)].mean)                                # impute E[correct | c]
      calibration[bin(c_e)] ← update(success, weight w)
```

## Design decisions and the alternatives considered

| Decision | Chosen | Alternatives considered | Why |
|---|---|---|---|
| Decision unit | Plan (tool set as a DAG) | Per-tool independent choice (B5); end-to-end set classifier (B2) | Dependencies make per-tool choice infeasible-prone. Ablation A4 measures this. |
| Objective | Constrained min-cost | Weighted sum of success, cost and latency (B5) | A weighted sum lets a cheap wrong answer win. Constraints match how SLOs are written. |
| Plan search | DFS branch-and-bound with an admissible bound (max over open types of cheapest provider) and most-constrained-type branching | ILP; beam search; exhaustive enumeration | Exact for the k cheapest within the candidate set, simple, and fast. A property test (`tests/property`) checks the optimum against brute force on 200 random registries. |
| Exploration | Thompson sampling with **global-time** discounting | UCB; ε-greedy; per-observation discounting | Global-time decay restores variance for unused tools, so recovered tools are re-explored. TS composes naturally with Monte-Carlo constraint checks. |
| Context | Family-level Beta with hierarchical pooling to the tool's global rate | Linear contextual bandits (LinTS) | Families are discrete and few. Pooling gives unseen or rare families a sensible prior. |
| Credit assignment | Soft responsibilities ∝ (1 − confidence) | Uniform 1/k blame; ignore failures | Uniform blame over-penalised good tools in multi-tool plans. With it, the calibration conflated other types' errors and the VoI rule over-verified (development seed: stationary cost/success 4.13 → 3.80 m$ after the change). |
| Superseded answers | Imputed with the calibrated P(correct \| confidence) (ablation A10) | Ignore them (v1) | When verification replaces an answer, only the replacement is graded. Ignoring the replaced answer hid exactly the mistakes verification caught: on a development seed, a tool with true quality 0.58 was estimated at 0.91. With imputation, success under drift is +1.5 points and calls are fewer, but cost per success is +0.38 m$, because ATLAS stops relying on cheap tools plus verification (E3). |
| Candidate scoring | Only providers of the needed types and their input closure | Score every registered tool | Same decisions, less work: planning p95 at 512 tools fell from 27 ms to 15.5 ms (E4). |
| Safe exploration | Sensitive tools use posterior means only | Exploring everything | Licensed or production-critical tools should not be used just to learn. |
| Stopping | Calibrated VoI, triggered only below the request's own Q_min | Fixed confidence threshold | A fixed threshold ignored the SLO and over-verified on the development seed. |

## Complexity

Let $m$ be the number of candidates kept per type, $|\hat Y^*|$ the size of the type closure, and $k$ the number of plans kept. Candidate
filtering is linear in the number of providers of the relevant types (not in the registry size), with posterior-utility sorting over providers per type. Branch-and-bound explores at most
$O(m^{|\hat Y^*|})$ nodes but prunes hard, because the bound uses per-type minimum costs. Scoring is
$O(k\cdot|P|\cdot S)$ for $S$ Monte-Carlo latency samples. E4 measures the end-to-end overhead as the registry grows.

## Known limitations

* The per-tool quality model is a product of independent Bernoullis. Correlated errors (two search engines citing the
  same wrong source) are not modelled.
* Credit assignment is heuristic. It needs a confidence signal that is informative about correctness.
* Task understanding is a small supervised text classifier. Accuracy on held-out phrasings is about 0.88 exact-set, and
  its errors cap every router that depends on it. E6 swaps in a local LLM and an oracle and sweeps understanding
  accuracy.
* The simulator's parameters are hand-specified. The live evaluation (`benchmarks/live`) runs real tools on generated
  data, with injected latency and faults.
* **Large registries of near-duplicate tools remain unsolved.** At 512 tools, success is 0.79 vs 0.91 at 12. Three
  remedies were tried on development seeds and did not help, so they were dropped (see `research/hypotheses.md`):
  * √N admission control (worse: it admits cheap duplicates before the genuine tools);
  * pooled type-level quality priors;
  * pessimistic constraints against the optimizer's curse.
