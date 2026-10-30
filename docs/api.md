# HTTP API

All `/v1` endpoints require `X-API-Key`. Keys map to roles through `ATLAS_API_KEYS="key:role,key:role"`.
Every response carries `X-Request-ID` (the client may supply it). Interactive OpenAPI docs are at `/docs`.

| Method | Path | Purpose |
|---|---|---|
| GET | `/healthz` | Liveness |
| GET | `/readyz` | Readiness (model trained, tools registered) |
| GET | `/metrics` | Prometheus metrics |
| GET | `/v1/tools` | Tools visible to the caller's role |
| POST | `/v1/route` | Plan only, for agents that execute tools themselves |
| POST | `/v1/execute` | Plan, execute and return the evidence |
| POST | `/v1/feedback` | Report whether the answer for `request_id` was correct |
| GET | `/v1/stats` | Learned per-tool estimates and circuit states (admin) |
| GET | `/ui` | Interactive console: plan, run, inspect and give feedback in a browser (calls the endpoints above with your API key) |

## Example

```bash
curl -s localhost:8000/v1/execute -H 'X-API-Key: dev-key' -H 'Content-Type: application/json' \
  -d '{"query": "Calculate the percentage change in revenue between Q1 and Q2",
       "constraints": {"latency_budget_ms": 1500, "min_quality": 0.8}}'
```

Response (abridged):

```json
{
  "decision": {
    "request_id": "6f1c…",
    "family": "revenue_growth",
    "required_types": ["internal_metrics", "numeric_result"],
    "plan": [
      {"tool": "sql_warehouse", "supplies": ["internal_metrics", "scalars", "table"], "depends_on": [], "purpose": "primary"},
      {"tool": "calculator", "supplies": ["numeric_result"], "depends_on": [0], "purpose": "primary"}
    ],
    "estimates": {"cost_usd": 0.0021, "success": 0.86, "p_latency_over_budget": 0.0},
    "constraints_satisfied": true,
    "rejected": {"python_sandbox": "pruned_top4:numeric_result"},
    "notes": ["constraints satisfied"]
  },
  "execution": {"latency_ms": 31.4, "cost_usd": 0.0021, "calls": [], "evidence": {"numeric_result": {"output": {"value": 7.0}}}}
}
```

Which plan is chosen depends on what the router has learned so far. Read the actual decision from the response;
don't hard-code the example above.

`arguments` (optional) supplies per-tool inputs, e.g. `{"sql_warehouse": {"sql": "SELECT …"}}`. Without it, each
adapter derives its input from the query (the demo SQL adapter has a small deterministic NL→SQL resolver). In an LLM
agent, the agent fills `arguments` for the tools ATLAS selected.

## Feedback loop

Reliability and latency are learned from every execution. Answer quality is learned only from
`POST /v1/feedback {"request_id": "...", "success": true|false}` (a user signal, an evaluator, or a downstream check).
Unacknowledged requests expire from a bounded buffer (10,000 entries).
