# FootballAi V2 local demo

## Prerequisites

- Python 3.13 virtual environment at `.venv-test`;
- dependencies installed from `requirements-test.txt`;
- Node.js with npm;
- dashboard dependencies installed automatically on the first demo start.

If the Python environment does not exist:

```bash
python3.13 -m venv .venv-test
.venv-test/bin/python -m pip install -r requirements-test.txt
```

## One command

From the repository root:

```bash
make v2-demo
```

The command:

1. checks the local Python and Node runtimes;
2. installs locked frontend packages when needed;
3. imports the committed V1 artifacts only when no local V2 run exists;
4. starts FastAPI on `127.0.0.1:8000`;
5. starts Vite on `127.0.0.1:5173`;
6. prints both URLs and stops both processes on Ctrl+C.

Open:

```text
http://localhost:5173
```

If port 5173 is already reserved by another local project, choose a free port
without stopping that project:

```bash
FOOTBALLAI_V2_DASHBOARD_PORT=5174 make v2-demo
```

No inference, V1 regeneration, cloud service, database, queue, container, or
deployment is started.

## Complete tests

Install the bounded browser once:

```bash
cd v2/dashboard
npx playwright install chromium
cd ../..
```

Then run:

```bash
make v2-test
```

Generated demo runs, builds, browser screenshots, reports, and dependency
directories are ignored local state.
