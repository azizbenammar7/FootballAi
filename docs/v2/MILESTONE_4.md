# V2 Milestone 4 — Local Read API

## Scope

Milestone 4 adds a local FastAPI adapter around the caller-configured
`LocalAnalysisRunStore`. It adds no database, queue, cloud service, deployment,
or write endpoint.

## Endpoints

```text
GET /api/health
GET /api/v1/runs
GET /api/v1/runs/{run_id}
GET /api/v1/runs/{run_id}/manifest
GET /api/v1/runs/{run_id}/artifacts
GET /api/v1/runs/{run_id}/summary
GET /api/v1/runs/{run_id}/players
GET /api/v1/runs/{run_id}/players/{player_id}
```

Run IDs use UUID-v4 path validation. Missing runs and tracks return safe 404
responses. Unsupported or corrupt legacy data returns a safe 422 response.

## Legacy response adapter

The adapter translates integrity-checked imported artifacts into stable team,
track-list, and track-detail responses. It provides distance summaries,
advisory distributions, approximate 15-minute blocks, heatmaps, speed and
estimated cumulative-distance timelines, and the **Workload and Fatigue
Advisory** breakdown.

Track IDs are always labelled `Legacy track <id>` or `Unverified player track
<id>` and `identity_verified` is false. Every legacy summary and track response
propagates the mandatory data-quality warnings. The API never silently treats a
temporary V1 track as an identified player.

## Safety controls

- The filesystem root is supplied by the application configuration.
- User input cannot select arbitrary filesystem paths.
- Artifact reads resolve registered IDs through the manifest and revalidate
  path containment, symlink safety, size limits, byte size, and SHA-256.
- `file://` and absolute input URIs are redacted from public responses.
- Typed response models reject accidental response-field expansion.
- CORS accepts only explicitly configured HTTP localhost origins.
- Every response includes a generated request ID and response timing.
- Structured request logs contain method, route, status, and duration without
  artifact content or internal paths.

## Local command

```bash
FOOTBALLAI_V2_RUN_ROOT=data/runs \
FOOTBALLAI_V2_CORS_ORIGINS=http://localhost:5173 \
PYTHONPATH=v2/src \
.venv-test/bin/python -m uvicorn footballai_v2.api.main:app \
  --host 127.0.0.1 --port 8000
```

Dependencies are pinned in `v2/requirements-api.txt`. API tests use temporary
run roots and a local ASGI test client; they require no network or external
service.
