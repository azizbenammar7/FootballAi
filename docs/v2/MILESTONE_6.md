# Milestone 6 — real analysis execution workflow

Milestone 6 changes the V2 local platform from a read-only run browser into an
end-to-end asynchronous workflow while retaining
`footballai.analysis-run/v1` as the sole run contract.

Delivered:

- streaming multipart upload with checksum, limits, filename controls, media
  type checks, and bounded `ffprobe` inspection;
- isolated `input/`, `artifacts/`, `logs/`, and `tmp/` run namespaces;
- provider-neutral queue boundary with a durable local filesystem adapter;
- separate restart-aware worker and persistent cancellation checkpoints;
- deterministic `demo_fast` and optional genuine `v1_compat` execution;
- weighted progress, retry, clone, safe failure, and immutable attempt chains;
- stable generated artifact schemas and integrity verification;
- New Analysis and progress pages with upload progress, cancellation
  confirmation, attempt navigation, results handoff, and persistent quality
  warnings;
- queue, upload, worker, API, component, build, and Playwright workflow tests.

Security boundaries include local-only CORS, no authentication-free remote
deployment, no arbitrary shell command strings, no user-derived paths, no
unbounded media defaults, no log endpoint, and Git ignores for inputs and
runtime state.

Not included: Azure, Terraform, databases, external queues, Docker,
Prometheus/Grafana, detector benchmarking, homography calibration, or a
production full-match V2 run. The next milestone should introduce a remote
deployment threat model, authentication, object storage, a distributed queue,
and production observability before exposing this API beyond localhost.
