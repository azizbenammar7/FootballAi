# AI3-B — pitch calibration in the analysis pipeline

AI3-B connects the AI3 calibration engine
([`AI3_PITCH_CALIBRATION.md`](AI3_PITCH_CALIBRATION.md)) to FootballAI's
asynchronous analysis pipeline. It also adds a first spatial feature: a
**tracklet-level pitch-space diagnostic**. This diagnostic is **not a verified
individual player heatmap**.

Code:
- `v2/src/footballai_v2/execution/ai3/pipeline_stage.py`: stage logic and the
  interval read model
- `v2/src/footballai_v2/execution/spatial_stages.py`: worker glue
- `v2/src/footballai_v2/execution/executor.py`: stage loop
- `GET /api/v1/runs/{run_id}/pitch-space`: API endpoint
- `v2/dashboard/src/components/PitchSpacePanel.tsx`: dashboard panel

Tests: `v2/tests/test_ai3b_pitch_pipeline.py`.

## Status

| Item | Status |
|---|---|
| Pipeline integration (`identity_resolution`, `pitch_calibration`) | Implemented. Tested on synthetic AI1 artifacts. |
| Spatial artifacts and their checksum bindings | Implemented. Tested. |
| Pitch-space diagnostic (API and dashboard) | Implemented. Tested on synthetic data. |
| **Calibration on real football footage** | **Not measured.** No real video, AI1 cache or annotated real frame was available in the development environment. No real-footage numbers are reported anywhere. |

## Flow

```text
POST /api/v1/analyses  (video + optional pitch_annotations file)
   │  checksum, probe, annotation checks (bound to this video's SHA-256)
   v
queued immutable attempt ── worker ──────────────────────────────────────────
   detection + tracking   YOLOv8m + ByteTrack (v1_compat): detections.parquet, tracklets.parquet
   identity_resolution    verify AI1 → AI2C review_only → player-identities (all UNKNOWN)
   pitch_calibration      verify AI1 + pinned annotations → AI3 calibrate every sampled frame
                          → project foot points on CALIBRATED frames only
                          → pitch-calibration, projected-player-observations, pitch-space-diagnostic
   metrics / workload / artifact_publication   (unchanged)
   │  every artifact must read back byte-for-byte before the run can succeed
   v
GET /api/v1/runs/{id}/pitch-space?start_seconds&end_seconds   → dashboard panel
```

The `v1_compat` pipeline version is now `v1_compat/2.2.0`. Both stages are
required for new attempts. Runs queued earlier keep their recorded stage
plan, so their two spatial stages stay skipped.

### AI1 verification before any spatial work

The checks are in `verify_ai1_artifacts`. Each failure has a coded,
non-retryable error.

| Check | Error code |
|---|---|
| The tracklets bytes match the checksum the tracker recorded | `ai1_tracklets_checksum_mismatch` |
| Detections were computed from this run's video | `ai1_video_binding_mismatch` |
| Tracklets were built from exactly these detections | `ai1_tracklets_binding_mismatch` |
| Every tracklet observation lies on an AI1 sampled frame with the same timestamp | `ai1_frame_alignment_mismatch` |
| Both parquet files can be read | `ai1_artifact_unreadable` |

### identity_resolution

AI2C runs in `review_only` mode, its default. It makes no automatic
association:
- every person tracklet is `unresolved`, so its identity is UNKNOWN
- non-person classes (for example `sports ball`) are `excluded`
- no team is assigned

This stage publishes `player-identities` (`footballai.player-identities/v1`).
Its only purpose is to give the projected observations a verified identity
provenance. No identity is invented.

### pitch_calibration

**Without annotations,** the stage succeeds with an explicit message:
- every AI1 sampled frame is `uncalibrated`, with reason
  `no_landmark_annotations`
- no pitch coordinate exists
- only `pitch-space-diagnostic` is published, with
  `calibration.state = not_attempted`

There is no pixel→metre fallback.

**With annotations,** the stage does the following:
1. It re-verifies the annotation SHA-256 pinned in the run parameters.
2. It checks the annotations against this run's video checksum, frame size
   and AI1 timestamps.
3. It calibrates every AI1 sampled frame with the AI3 policy
   (`ai3-calibration-policy/v1`). Suspected cuts come from the tracklet
   birth-burst rule, which can only remove calibration.
4. It projects AI2C observations through `project_player_observations`.

It publishes:

| Artifact ID | Contract | Content |
|---|---|---|
| `pitch-calibration` | `footballai.pitch-calibration/v1` | Self-verifying calibration, bound to the video, detections, tracklets and annotation checksums |
| `projected-player-observations` | `footballai.projected-player-observations/v1` (Parquet) | One row per eligible tracklet observation: projection status, `pitch_x_m`/`pitch_y_m` only when `projected`, inside-lines and extrapolation flags, frame calibration state, source keyframe. Metadata names every source checksum. |
| `pitch-space-diagnostic` | `footballai.pitch-space-diagnostic/v1` | Label, evidence status, calibration summary and keyframe metrics, pitch geometry, per-frame state and reason, observation counts by projection status, identity counts, evidence checksums, limitations |

If the annotations do not match the run's frames, frame size or checksums,
the attempt fails with `pitch_annotations_incompatible`. The failure is
recorded as non-retryable and nothing is published.

### Failure, retry and idempotency

- **Non-retryable failures.** `ExecutionFailure.retryable` is now recorded on
  the run's failure. Deterministic refusals, such as incompatible annotations
  or AI1 binding errors, are marked non-retryable.
- **Retry and clone.** A retry is a new immutable attempt, and it keeps the
  pinned annotations. `POST /api/v1/runs/{id}/clone` can attach new
  annotations, which replace any the source run carried. They must be bound
  to the same video.
- **Redelivery.** If a worker crashes after the spatial stages but before
  publication, the redelivered job rebuilds the AI1 payloads and recomputes
  the spatial artifacts (`SpatialStages.ensure`).
- **Deterministic outputs.** The calibration and observation bytes depend
  only on their inputs. There is no wall-clock time and no unseeded
  randomness. The test `test_identical_attempts_publish_identical_spatial_bytes`
  pins this for the calibration artifact.

## Attaching annotations to a run

Annotation upload rules:
- The `pitch_annotations` multipart field is accepted on
  `POST /api/v1/analyses` and `POST /api/v1/runs/{id}/clone`.
- It is only accepted for the `v1_compat` profile.
- It must be at most 1 MiB.
- The browser direct-to-object upload path does not accept annotations in
  this release.

The upload is refused before any run is created when:

| Code | Cause |
|---|---|
| `invalid_pitch_annotations` | Not valid `footballai.pitch-landmark-annotations/v1` |
| `pitch_annotations_unbound` | `source.video_sha256` is missing |
| `pitch_annotations_video_mismatch` | Made for another video |
| `pitch_annotations_bound_to_ai1` | Declares detections or tracklets checksums. AI1 bytes change with every attempt (tracklets name their run), so such a file could never match. Create the file with `template --bind video`. |
| `pitch_annotations_synthetic` | Synthetic annotations on a non-synthetic analysis |
| `pitch_annotations_no_keyframes` | No annotated frame |
| `pitch_annotations_unsupported_profile` | Profile other than `v1_compat` |
| `pitch_annotations_too_large` | More than 1 MiB (HTTP 413) |

The canonical bytes are stored in the run parameters as
`pitch_annotations = {schema, sha256, uploaded_sha256, size_bytes, origin, keyframes, document}`.
The worker re-serialises the document and refuses to calibrate if its SHA-256
differs (`pitch_annotations_checksum_mismatch`).

## Pitch-space diagnostic

```text
GET /api/v1/runs/{run_id}/pitch-space?start_seconds=&end_seconds=&max_points=
```

The endpoint reads `pitch-space-diagnostic` and, when present,
`projected-player-observations`. Both reads check the registered size and
SHA-256. The response is `footballai.pitch-space-view/v1` and contains:

- `label`: the "not a verified individual player heatmap" wording,
  `level: tracklet`
- `evidence_status`: `no_annotations`, `manual_annotations` or
  `synthetic_annotations`, plus `data_origin`
- `interval`: the AI1 sampled frames in the interval:
  - counts by state (calibrated, uncalibrated, invalidated)
  - counts by reason
  - calibrated fraction
  - contiguous state segments with source timestamps and frame indices
- `observations`: counts by projection status, projected, distinct
  tracklets, extrapolated, inside the lines
- `occupancy`: projected tracklet observations per 5 m cell, over the pitch
  plus a 5 m margin, in `footballai.pitch-plan/v1` metres. It is `null`
  without calibration.
- `points`: up to `max_points` projected foot points (default 2000, at most
  5000). Each carries its frame index, timestamp, tracklet ID, identity
  status and source keyframe. When there are more, an evenly spaced
  subsample is returned.
- `evidence` and `artifacts`: every source checksum and the verified
  artifact references
- `limitations`

The dashboard panel on the run page draws:
- the plan view in metres, with markings from the configured geometry
- the occupancy cells and the points (extrapolated points are hollow)
- a calibration timeline

It also shows coverage, missing and invalidated counts, and the reasons
frames have no position, and it has a time-interval filter. When there is no
calibration, it says so and draws no pitch.

What the diagnostic does **not** do:
- It does not aggregate per player.
- It does not assign teams.
- It does not separate referees or staff.
- It does not compute distance, speed or displacement.
- It does not use, or replace, the legacy V1 "heatmap" fields.

## Real football validation: exact local workflow

No measurement in this section has been made. Run it locally, on licensed
footage, with explicit approval for the inference it needs. Keep frames,
pixels and video outside Git. `annotate-frame` refuses to write inside a Git
tree.

```bash
export PYTHONPATH=v2/src
M="python -m footballai_v2.execution.ai3.cli"
PITCH="--length-m <measured length> --width-m <measured width> --markings ifab-law-1 --dimensions-source '<source of the measurement>'"
```

1. **AI1 on a bounded clip.**
   1. Upload the clip with the `v1_compat` profile and no annotations. This
      runs YOLOv8m + ByteTrack, so it needs approval.
   2. The run publishes `detections.parquet` (local backend:
      `$FOOTBALLAI_V2_RUN_ROOT/<run_id>/artifacts/detections.parquet`) and an
      all-uncalibrated pitch-space diagnostic. The diagnostic's
      `frame_states` lists every sampled frame index and timestamp.
2. **Choose shots and keyframes.**
   - Declare shots with their real camera motion (`static`, `moving` or
     `unknown`) and camera side.
   - For every shot to be measured, pick at least two keyframes, separated in
     time, and preferably including frames where the camera has panned.
   - Hold out at least one shot entirely for evaluation.
3. **Make a video-bound template:**

   ```bash
   $M template $PITCH --source-id <clip> --detections <run>/artifacts/detections.parquet \
     --bind video --shot s1:<start>:<end>:<static|moving|unknown>:<side> \
     --keyframe <index> --keyframe <index> --output ~/footballai-annotations/<clip>.json
   ```

4. **Click landmarks for each keyframe:**

   ```bash
   $M annotate-frame $PITCH --video <clip.mp4> --frame-index <index> --timestamp <t> \
     --shot-id s1 --output-dir ~/footballai-frames
   ```

   - Aim for 8–10 or more visible landmarks.
   - Mark about one in three as `validation`. Spread the validation points
     beyond the fit hull.
   - A second annotator should repeat a subset of frames, to measure click
     agreement.
5. **Review.** A reviewer sets each point's `review_status` to `accepted` or
   `rejected`.
6. **Measure offline.** These commands run no inference:

   ```bash
   $M validate  --annotations <clip>.json --detections <det.parquet>
   $M calibrate --annotations <clip>.json --detections <det.parquet> --tracklets <trk.parquet> \
     --suspected-cuts-from-tracklets --output ~/footballai-annotations/<clip>-calibration.json
   ```

7. **Run through the pipeline.** This re-runs AI1, so it needs approval:

   ```bash
   curl -F pitch_annotations=@<clip>.json http://localhost:8000/api/v1/runs/<run_id>/clone
   ```

   Then read `GET /api/v1/runs/<new_run_id>/pitch-space`.

Record the following in a results table, per shot and in total:

| Measure | Source | Value |
|---|---|---|
| Calibration coverage (calibrated / analysed frames) | `frames.by_status`, `calibrated_fraction` | not measured |
| Held-out reprojection error, median / max (validation landmarks, calibrated keyframes) | `calibration.keyframes[].metrics.validation_*`, `summary.validation_error_*` | not measured |
| Invalidated frames, by reason | `frames.by_reason` | not measured |
| Camera movement limitations (frames lost to `camera_not_declared_static`, `outside_verified_keyframe_interval`, `static_shot_keyframes_disagree`, `suspected_camera_cut`) | `frames.by_reason` | not measured |
| Projection uncertainty: share of projected points extrapolated beyond the fit landmarks, and inter-annotator click agreement | `observations.projected_extrapolated_beyond_landmarks`, second-annotator comparison | not measured |

Held-out reprojection error measures calibration on annotated landmarks only.
It is not player-position accuracy, which also includes the foot-point bias
of box bottoms (not corrected).

## Known limitations

- **Real accuracy is unmeasured.** The synthetic benchmark (AI3) shows that
  2–6 validation landmarks cannot bound extrapolation error.
- **Annotations are manual.** There is no automatic pitch keypoint detector.
  Moving-camera footage is calibrated only on annotated keyframes.
- **Tracklets fragment.** Referees, staff and other persons appear as
  UNKNOWN tracklets.
- **Projected observations are stored as rows.** A full match produces a few
  hundred thousand rows in one Parquet file. The read model loads it in the
  API process, which is adequate for a diagnostic but not for high request
  rates.
- **Direct upload is not supported.** The browser direct-to-object upload
  path cannot attach annotations yet.
