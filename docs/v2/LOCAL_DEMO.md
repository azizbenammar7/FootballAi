# FootballAi V2 local demo

For the **portfolio Tier-A rehearsal**, use the isolated [M4-A launcher and recording guide](../portfolio/M4A_DEMO_PACKAGE.md). It creates a fresh generated `demo_fast` run, skips legacy import, enables the offline scripted agent, forces local adapters, and binds both services to `127.0.0.1`. It never uses private footage or a paid model.

From the repository root:

```bash
python3.13 -m venv .venv-test
.venv-test/bin/python -m pip install -r requirements-test.txt
make v2-demo
```

The command installs locked dashboard packages when absent, imports the
committed legacy result example, creates one tiny generated `demo_fast`
workflow when needed, and starts three child processes: FastAPI, the filesystem
queue worker, and Vite. It prints the selected dashboard/API ports plus the run
and queue roots. Ctrl+C terminates all three.

The preferred ports are 5173 and 8000. Occupied ports are not killed; the demo
selects a free port in the next bounded range. Overrides are supported:

```bash
FOOTBALLAI_V2_DASHBOARD_PORT=5174 \
FOOTBALLAI_V2_API_PORT=8001 \
FOOTBALLAI_V2_RUN_ROOT=/tmp/footballai-runs \
FOOTBALLAI_V2_QUEUE_ROOT=/tmp/footballai-queue \
make v2-demo
```

The generated video is a one-second solid-color evaluation fixture created
locally with FFmpeg. It contains no match footage and is never committed.
`demo_fast` values are deterministic synthetic workflow output, not match
measurements. `v1_compat` only becomes selectable when its optional local CV
dependencies and existing model weights are present; it never downloads them
as part of tests.

To prepare and start the optional preserved-algorithm profile:

```bash
make v2-v1-compat-setup
make v2-v1-compat-readiness
make v2-demo-v1-compat
```

The setup uses `.venv-test`, prepares ignored `.models/yolov8m.pt`, and is the
only command allowed to download that model when absent. API, worker, seed,
and readiness processes all use the same `FOOTBALLAI_V2_PYTHON` executable.
The combined demo refuses to start until readiness is `ready`, preserves
user-provided run/queue/model/device overrides, and prints device, model, API,
and dashboard information. See [V1_COMPAT_SETUP.md](V1_COMPAT_SETUP.md).

## Copilot question flow

The copilot is off by default, and `make v2-demo` does not enable it. Bind to
`127.0.0.1` only: the application has no authentication.

```bash
FOOTBALLAI_COPILOT_ENABLED=1 FOOTBALLAI_ENVIRONMENT=local PYTHONPATH=v2/src .venv-test/bin/python -m uvicorn footballai_v2.api.main:app --host 127.0.0.1 --port 8000
```

```bash
cd v2/dashboard && npm run dev
```

Open a completed run under **Analysis runs**, then ask these in the panel:

| Ask | `demo_fast` run | Local `v1_compat/1.0.0` run | A `v1_compat/2.x` run |
|---|---|---|---|
| "Explain this analysis run." | Answered, labelled synthetic | Answered, partial | Answered |
| "What do the tracking diagnostics tell us?" | Says nothing was measured | Track ID and detection counts, partial | Full fragmentation diagnostics |
| "How many successful passes did player 7 make?" | Not answerable | Not answerable | Not answerable |

Local `v1_compat` runs correctly show **Real inference not verified**, because they
predate `/v2` diagnostics. Do not edit run metadata to change that. A
`verified_real` answer needs a fresh licensed run (M0). The badge shows the offline
deterministic mode unless a real model was explicitly enabled.

### Read-only agent

The agent has its own flag, and answers only on this machine. Replace the API command
with this one. The dashboard's origin must be allowed, which the default is.

```bash
FOOTBALLAI_COPILOT_ENABLED=1 FOOTBALLAI_COPILOT_AGENT_ENABLED=1 FOOTBALLAI_ENVIRONMENT=local PYTHONPATH=v2/src .venv-test/bin/python -m uvicorn footballai_v2.api.main:app --host 127.0.0.1 --port 8000
```

Choose **Read-Only Agent** in the panel. It is labelled `OFFLINE SCRIPTED AGENT` because a
fixed script chooses the tools. Never describe it as a language model. These settings
cannot reach a paid model. That needs a separate flag, two caps, and an approved plan: see
[COPILOT_AGENT_REAL_MODEL.md](COPILOT_AGENT_REAL_MODEL.md).

| Ask | A `v1_compat/2.x` run | Tool trace |
|---|---|---|
| "What do the tracking diagnostics tell us?" | Answered | `get_tracking_diagnostics` |
| "What model was used in the analysis?" | Answered | `get_model_provenance` |
| "What evidence is missing?" | Says nothing was recorded missing, or states the gap | `get_evidence_limitations` |
| "How many successful passes did player 7 make?" | Not answerable | No tool ran |

An unverified run gives a different result for "What evidence is missing?": it stays
`evidence_not_available` unless a gap was recorded. To use other ports, set
`FOOTBALLAI_V2_CORS_ORIGINS` to the dashboard's exact origin, or the route answers `404`.

Complete validation:

```bash
make v2-test
```

Before any remote deployment, authentication, remote object storage, a
distributed queue, and deployment-specific resource controls are mandatory.
