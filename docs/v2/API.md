# V2 local API

All responses are typed by FastAPI, reject extra fields, include a request ID,
and avoid absolute filesystem paths. CORS permits configured HTTP localhost
origins only.

| Method | Endpoint | Purpose |
|---|---|---|
| GET | `/api/health` | Local service and contract health |
| GET | `/api/v1/pipeline-profiles` | Availability, requirements, warnings, and GPU role |
| POST | `/api/v1/analyses` | Stream, probe, store, and enqueue one multipart video |
| GET | `/api/v1/runs` | List immutable attempts |
| GET | `/api/v1/runs/{run_id}` | Attempt details, safe provenance, stages, and chain |
| GET | `/api/v1/runs/{run_id}/progress` | Weighted progress and allowed controls |
| POST | `/api/v1/runs/{run_id}/cancel` | Cancel queued work or persist a running request |
| POST | `/api/v1/runs/{run_id}/retry` | New linked attempt from failed/partial |
| POST | `/api/v1/runs/{run_id}/clone` | New logical analysis from uploaded bytes |
| GET | `/api/v1/runs/{run_id}/manifest` | Public redacted manifest |
| GET | `/api/v1/runs/{run_id}/artifacts` | Registered artifact metadata and integrity |
| GET | `/api/v1/runs/{run_id}/summary` | Team summary |
| GET | `/api/v1/runs/{run_id}/players` | Unverified track summaries |
| GET | `/api/v1/runs/{run_id}/players/{player_id}` | Track detail and advisory |

`POST /api/v1/analyses` accepts `video`, required `match_name`, and optional
`home_team`, `away_team`, `competition`, `match_date`, `venue`, `notes`,
`data_origin`, and `pipeline_profile`. Strings and upload resources are
bounded. Expected validation failures return safe 4xx details; tracebacks,
ffprobe output, and local paths are not public.
