# AI1 — detection/tracking decoupling

AI1 separates expensive detector inference from temporary tracking identity. It
does **not** solve persistent player identity.

## Architecture

Before AI1:

```text
video -> coupled YOLO + ByteTrack -> filtered track fragments
```

After AI1:

```text
video
  -> YOLO detection (once)
  -> footballai.detections/v1 (Parquet)
  -> ByteTrack over cached detections (independently rerunnable)
  -> footballai.tracklets/v1 (Parquet)
  -> footballai.analysis-diagnostics/v2
  -> preserved V1 compatibility metrics/advisory adapter
```

The detection artifact stores every sampled frame, including frames with zero
detections. This is required for exact tracker replay because ByteTrack ages
lost tracks on empty frames. Detection rows contain frame/time, semantic class,
confidence, and pixel bounding boxes. Parquet metadata records model family,
filename and runtime version, model checksum, confidence, image size,
target/effective/source FPS, sampling stride, source dimensions/frame count and
duration, device, source-video checksum, code revision, and pipeline version.
No local model or video path is published.

The tracklets artifact stores every temporary tracking observation with its
track ID, frame/time, pixel bounding box, centroid-derivable coordinates,
confidence, and class. Its metadata records the ByteTrack version and complete
configuration plus the source run and the exact detections artifact checksum.
Tracklet start/end, duration, observation count, and confidence summaries are
deterministically derivable and exposed through the typed reader.

`raw_tracks.parquet` remains a private compatibility input for the preserved
statistics scripts. The adapter retains the historical 10-observation filter;
the AI1 tracklets artifact does not discard shorter fragments. The later
historical 50-observation threshold remains unchanged.

## Descriptive diagnostics

`footballai.analysis-diagnostics/v2` separates `observed_metrics` from
`interpretation`. It reports detection/tracklet counts, duration and observation
distributions, documented 1/5/10-second short-fragment rates, the existing
50-observation insufficiency rate, tracklets/starts/ends per video minute,
observed simultaneous tracked objects, detection-assignment fraction, and
five-minute fragmentation trend bins. Detection, tracking, and diagnostic
computation runtimes are recorded separately.

These are fragmentation proxies, not labelled identity-quality measurements.
The artifact explicitly reports ground-truth metrics as `not_measured` and does
not claim any identity metric that requires annotations.

## Cached tracker rerun

The tracking CLI accepts only a detections artifact, tracker configuration, and
source-run identifier. It has no video or model argument:

```bash
PYTHONPATH=v2/src .venv-test/bin/python -m \
  footballai_v2.execution.adapters.v1_compat_tracking \
  --detections /path/to/detections.parquet \
  --tracker pipeline/bytetrack_custom.yaml \
  --source-run-id 00000000-0000-4000-8000-000000000001 \
  --output-dir /tmp/footballai-tracker-baseline
```

Copy the YAML to a new bounded configuration, choose a new output directory,
and repeat the command to compare fragmentation proxies against the same cache.
No detector process is imported or invoked by this command. Experiments on a
single clip are exploratory and do not establish a universally better tracker.

## Boundary for AI2

AI2 may consume the immutable detections and tracklets artifacts through their
typed readers and add a separate identity resolver. AI1 deliberately includes
no ReID embeddings, team assignment, jersey recognition, identity merge UI,
pitch calibration, ball tracking, possession, events, tactical metrics, or
coach-facing LLM insight. A `tracklet` is a temporary tracker result, never a
player identity.
