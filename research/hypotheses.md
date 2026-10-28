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

**Expected trade-off (stated in advance).** The VoI rule spends extra calls to buy success, so ATLAS is expected to make
*more* calls per task and *more* strict-unnecessary calls than routers that never verify. That is a deliberate,
parameterised trade-off (value of success $V$, quality target $Q_{\min}$), and E5 maps it. If verification did not pay for
itself in cost per success, H2 would fail and we would report that.
