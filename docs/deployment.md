# Deployment

## Local (no containers)

```bash
python -m venv .venv && . .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
uvicorn atlas.api.app:app --port 8000
```

On first start the service creates the demo warehouse (`data/warehouse.db`) and trains the task-understanding model
(about 1 s). Learned state is snapshotted to `data/state/router.json` on shutdown and restored on start.

## Docker Compose (API + Redis + Prometheus)

```bash
ATLAS_API_KEYS="change-me:admin,analyst-key:analyst" docker compose up --build
curl localhost:8000/readyz
open http://localhost:9090          # Prometheus, scraping api:8000/metrics
```

| Setting | Meaning | Default |
|---|---|---|
| `ATLAS_API_KEYS` | `key:role` pairs, comma separated | `dev-key:admin` (**development only**) |
| `ATLAS_REDIS_URL` | Router-state snapshots in Redis instead of a file | unset |
| `ATLAS_STATE_PATH` | Snapshot file when Redis is not used | `data/state/router.json` |
| `ATLAS_DATA_DIR` | Corpora and warehouse location | `./data` |

## Hardening already in place

* The container runs as UID 10001 with a read-only root filesystem, `cap_drop: ALL` and `no-new-privileges`. Only
  tmpfs paths are writable.
* SQL: read-only file handle, an engine authorizer that allows only reads, a single statement, a row cap and a time
  budget.
* Python sandbox: a static AST allow-list, a separate isolated interpreter (`-I -S`), a hard timeout and an output cap.
  **This is not a full sandbox.** For untrusted code in production, run the sandbox in its own container with
  gVisor/nsjail, no network, and CPU/memory cgroup limits.
* Tool outputs are returned as data and never interpreted as instructions by the router. Prompt-injection defences
  for the *agent* that reads the evidence are out of scope.

## Operational signals

`atlas_decisions_total{feasible}` (rising `false` = SLOs unattainable with current tools),
`atlas_tool_calls_total{tool,outcome}`, `atlas_tool_latency_seconds`, `atlas_routing_seconds`,
`atlas_request_cost_usd`, `atlas_feedback_total`.
