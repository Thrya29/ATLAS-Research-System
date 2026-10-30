# Live evaluation (real tool execution)

Seeds 1–10 (10), 400 tasks each. Mean [95% bootstrap CI over seeds].

| Router | success | cost/success (m$) | outage success | p95 latency (ms) | calls/task |
|---|---|---|---|---|---|
| **ATLAS** | 0.902 [0.889, 0.916] | 3.327 [3.245, 3.417] | 0.809 [0.772, 0.848] | 1243 [1201, 1288] | 1.567 [1.538, 1.594] |
| ATLAS v1 | 0.901 [0.888, 0.914] | 3.309 [3.230, 3.392] | 0.806 [0.770, 0.842] | 1164 [1152, 1175] | 1.586 [1.555, 1.612] |
| B4 Static cost-aware | 0.786 [0.752, 0.820] | 3.458 [3.305, 3.601] | 0.514 [0.436, 0.591] | 1223 [1206, 1238] | 1.370 [1.345, 1.396] |
| B5 Contextual bandit | 0.516 [0.501, 0.534] | 2.577 [2.512, 2.632] | 0.456 [0.436, 0.474] | 584 [549, 615] | 0.622 [0.606, 0.639] |
| Rules (best tools) | 0.811 [0.784, 0.840] | 3.080 [3.024, 3.135] | 0.514 [0.442, 0.591] | 1269 [1202, 1339] | 1.352 [1.323, 1.380] |
| Rules (cheapest tools) | 0.484 [0.473, 0.494] | 1.138 [1.111, 1.167] | 0.469 [0.446, 0.493] | 74 [72, 76] | 1.418 [1.397, 1.437] |

Paired differences, ATLAS minus baseline (Holm-corrected Wilcoxon p over seeds):

| Baseline | Δ success | Δ cost/success (m$) |
|---|---|---|
| ATLAS v1 | +0.002 [+0.000, +0.003] (p=0.375) | +0.02 [-0.02, +0.07] (p=0.570) |
| B4 Static cost-aware | +0.116 [+0.089, +0.146] (p=0.020) | -0.13 [-0.24, +0.00] (p=0.375) |
| B5 Contextual bandit | +0.386 [+0.365, +0.404] (p=0.020) | +0.75 [+0.67, +0.85] (p=0.020) |
| Rules (best tools) | +0.091 [+0.070, +0.111] (p=0.020) | +0.25 [+0.18, +0.32] (p=0.020) |
| Rules (cheapest tools) | +0.419 [+0.405, +0.431] (p=0.020) | +2.19 [+2.11, +2.28] (p=0.020) |

Success by task family (mean over seeds):

| Router | kpi_lookup | growth | stats | policy | incident |
|---|---|---|---|---|---|
| ATLAS | 0.868 | 0.794 | 0.896 | 0.959 | 0.996 |
| ATLAS v1 | 0.867 | 0.796 | 0.898 | 0.959 | 0.988 |
| B4 Static cost-aware | 0.701 | 0.641 | 0.737 | 0.958 | 0.913 |
| B5 Contextual bandit | 0.649 | 0.000 | 0.000 | 0.954 | 0.984 |
| Rules (best tools) | 0.697 | 0.699 | 0.718 | 0.958 | 1.000 |
| Rules (cheapest tools) | 0.682 | 0.691 | 0.000 | 0.519 | 0.539 |

Measured real execution time per tool (excluding injected latency):

| Tool | calls | median (ms) | p95 (ms) |
|---|---|---|---|
| calculator | 3121 | 0.38 | 2.51 |
| docs_archive | 893 | 1.27 | 4.56 |
| docs_hybrid | 3838 | 2.11 | 7.51 |
| log_recent | 1232 | 1.11 | 4.10 |
| log_scan | 3721 | 2.67 | 11.50 |
| python_sandbox | 3978 | 123.85 | 555.19 |
| sql_primary | 7024 | 2.14 | 8.13 |
| sql_replica | 4371 | 1.95 | 6.12 |
