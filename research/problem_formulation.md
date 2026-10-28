# Problem formulation

## Objects

* Capability types $\mathcal{Y}$ (e.g. `internal_metrics`, `table`, `numeric_result`, `web_facts`).
* Tools $T$. Each $t$ has $\text{prov}(t) \subseteq \mathcal{Y}$, $\text{req}(t) \subseteq \mathcal{Y}$, a cost $c_t$, a
  role set $\rho_t$ and a sensitivity flag.
* A query $q$ with runtime context $x = (\text{role}, L_{\max}, Q_{\min}, \epsilon, \delta, V)$.
* The task's required types $Y(q) \subseteq \mathcal{Y}$ (unknown; estimated by $\hat{Y}(q)$) and its family $f(q)$ (the context).

## Feasible plans

A plan is a tool set $P \subseteq T$, executed as the DAG induced by its type dependencies. $P$ is feasible for
$(\hat{Y}, x)$ when

$$\hat{Y} \;\cup \bigcup_{t\in P}\text{req}(t) \;\subseteq\; \bigcup_{t \in P}\text{prov}(t), \qquad
\text{role}\in\rho_t\ \ \forall t\in P,\qquad \text{circuit}(t)\neq\text{open}\ \ \forall t\in P,$$

and the induced dependency graph is acyclic. $\mathcal{F}(q, x)$ denotes the feasible set. Types are OR-nodes (any
provider suffices) and tools are AND-nodes (all of their inputs are needed), which makes $\mathcal{F}$ the solution
set of an AND–OR graph.

## Outcome model

At time $\tau$, tool $t$ has a success (non-failure) probability $r_t(\tau)$, a log-normal latency
$\ell_t \sim \text{LN}(\mu_t(\tau), \sigma_t^2)$, and a family-specific correctness probability $\theta_{t,f}(\tau)$. With
$k$ retries:

$$R(P) = \prod_{t\in P}\bigl(1-(1-r_t)^{k+1}\bigr),\qquad Q(P) = \prod_{t\in P}\theta_{t,f},\qquad
\mathbb{E}[C(P)] = \sum_{t\in P} c_t\sum_{i=0}^{k}(1-r_t)^i,$$

and $L(P)$ is the critical-path length of the DAG under sampled latencies.

## Optimisation problem (per query)

$$\min_{P\in\mathcal{F}(q,x)} \ \mathbb{E}[C(P)] \quad \text{s.t.}\quad Q(P)\ge Q_{\min},\quad R(P)\ge 1-\epsilon,\quad
\Pr[L(P)>L_{\max}]\le\delta .$$

If no plan satisfies the constraints, choose $\arg\max_P R(P)\,Q(P)\,(1-\Pr[L(P)>L_{\max}])$ (expected on-time
success) and flag the decision.

The parameters $r, \mu, \sigma, \theta$ are unknown and non-stationary. Over a stream $q_1, q_2, \ldots$ the router must
learn them from **semi-bandit feedback**: per-call success and latency for every tool it ran, but only a task-level
success signal for answer quality.

## Stopping (sufficiency)

After the plan runs, evidence $e$ for type $y$ arrives with a confidence $c_e$. Let $\hat p(c_e)$ be the calibrated
probability that the evidence is correct. If $\hat p(c_e) < Q_{\min}$, the router may call an alternative provider $a$
when

$$\underbrace{(1-\hat p)\,r_a\theta_{a,f}\,V}_{\text{expected value of the extra evidence}} \;>\; c_a + \lambda V\frac{\mathbb{E}[\ell_a]}{L_{\max}}
\quad\text{and}\quad \mathbb{E}[\ell_a] \le L_{\max}-L_{\text{elapsed}} .$$

## Evaluation metrics

Task success; cost per successful task, $\sum C / \#\text{success}$; tool calls per task; strict unnecessary calls per task
(a call that could not supply a still-missing needed type, *including* verification calls); latency p50/p95/p99;
P(latency > budget); adaptation (success per drift window, share of the degraded tool, tasks to recover); calibration
(Brier score of $R\cdot Q$).
