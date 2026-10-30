# Hypotheses and protocol

**Protocol.** All development and tuning used seed 0 only. The hypotheses below, the hyper-parameters
(`ATLASConfig` defaults) and the experiment matrix were fixed *before* the reported runs, which use fresh seeds 1–20
(E1–E3) or the first 5/10 of them (E4/E5). Tests are paired on seeds, using common random numbers (identical tasks and
identical per-(task, tool, attempt) outcomes for every router). Each table applies a Holm correction.

| ID | Hypothesis | Primary metric | Comparison |
|---|---|---|---|
| H1 | ATLAS achieves higher task success than the rule-based router and the model-based (supervised) router. | success | ATLAS vs B1, B2 (E1, E2) |
| H2 | ATLAS has lower cost per successful task than the model-based router and the rule-based router. | cost / success | ATLAS vs B1, B2 (E1, E2) |
| H3 | Under drift, ATLAS beats the static cost-aware router (same planner, no adaptation), i.e. the gain is not just "using history". | success in drift windows | ATLAS vs B4 (E2) |
| H4 | Under drift, ATLAS beats a contextual bandit with a scalarised reward and no plan structure. | success, cost / success | ATLAS vs B5 (E2) |
| H5 | Dependency-aware plan search and the VoI stopping rule each contribute measurably. | success | ATLAS vs A4, A5 (E3) |
| H6 | Routing overhead stays within a few milliseconds per decision as the registry grows to 500 tools. | planning p95 | E4 |

### Follow-up after H6 failed (written 2026-10-29, before any run on seeds 1–20)

H6 failed: with 500 distractor tools, success fell from 0.90 to 0.78 and planning p95 rose to 27 ms. On the development
seed, 85% of plans used a near-duplicate "mirror" tool, and two thirds of routing time went to scoring every tool in the
registry.

**Registered first (H7a–c):** many-armed admission control (only $m + c\sqrt{N_t}$ tools per type compete) and
relevance-restricted scoring. H7a predicted that, at 512 tools, admission would raise success and keep the loss vs 12
tools within 0.05. H7b predicted a planning p95 of at most 10 ms. H7c predicted no change at 12 tools.

**What development showed (dev seeds 0, 101, 102 only; never seeds 1–20):**

* **H7a is not supported, and admission control was dropped.** It *lowered* success at 512 tools (0.754 vs 0.775).
  Ordering admissions by prior utility admits cheap mirrors before the genuine tools, which cost more and claim the same
  quality.
* Two further remedies did not help at 512 tools, so both were dropped:
  * Pooling quality priors across tools of a type: 0.767.
  * Pessimistic (lower-credible-bound) constraints against the optimizer's curse: 0.781.
* **The diagnosis that did hold up:** ATLAS's quality estimates were biased upward. Example: a mirror with true quality
  0.58 was estimated at 0.91 after 14 observations. The executor keeps only the most confident answer per type, so an
  answer later superseded by verification was never graded or learned from. The VoI rule hid exactly the mistakes it
  caught. The fix (**A10, imputation**) updates a superseded answer with the calibrated probability that an answer of
  its confidence is correct. On the dev seeds it raised success under drift at 12 tools (0.867 → 0.887), at a higher
  cost per success (4.31 → 4.73 m$). At 512 tools the gain was small (0.775 → 0.784).
* Relevance-restricted scoring is kept. It does not change decisions; it only avoids scoring irrelevant tools.

**Registered for the 20-seed run (ATLAS = v1 + imputation + relevance-restricted scoring):**

| ID | Hypothesis | Primary metric | Comparison |
|---|---|---|---|
| H7b | At 512 tools, planning p95 is at most 10 ms. | planning p95 | E4 |
| H8 | Imputing superseded answers raises success under drift. | success | ATLAS vs A10 (E3) |
| H9 | It also raises cost per success (the dev-seed trade-off reappears). | cost / success | ATLAS vs A10 (E3) |

**Expected trade-off (stated in advance).** The VoI rule spends extra calls to buy success, so ATLAS is expected to make
*more* calls per task and *more* strict-unnecessary calls than routers that never verify. That is a deliberate,
parameterised trade-off (value of success $V$, quality target $Q_{\min}$), and E5 maps it. If verification did not pay for
itself in cost per success, H2 would fail and we would report that.
