# FootballAi V2 local demo

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

Complete validation:

```bash
make v2-test
```

Before any remote deployment, authentication, remote object storage, a
distributed queue, and deployment-specific resource controls are mandatory.
