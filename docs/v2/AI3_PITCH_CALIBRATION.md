# AI3 — pitch calibration, homography and spatial foundation

AI3 converts image pixels into pitch coordinates in metres **only for frames
whose calibration is backed by evidence**. Every analysed frame is
`calibrated`, `uncalibrated` or `invalidated`. Only `calibrated` frames carry
a homography. Nothing falls back to a fixed pixel→metre scale: the V1
constant (≈0.086 m/px in `pipeline/02_stats.py`) is not used anywhere in AI3.

Code: `v2/src/footballai_v2/execution/ai3/`. Tests:
`v2/tests/test_ai3_pitch_calibration.py`. Evidence: `evaluation/ai3/`.

## Status in one paragraph

The contracts, the manual-landmark workflow, the estimator, the quality gates,
the camera-cut policy and the AI2C projection path are implemented and tested.
They have been validated on **synthetic** cameras only. No real football frame
has independent pitch annotations yet, so **calibration accuracy on real
broadcast footage is unmeasured**. The tooling to create those annotations is
included (see [Annotation workflow](#annotation-workflow)).

## Flow

```text
AI1 sampled frames / tracklets ──────────────┐
                                             │ frame indices, timestamps, checksums
manual landmark clicks (annotation file) ────┤
  roles fixed by the annotator: fit | validation
                                             v
              per keyframe: RANSAC homography from FIT landmarks only
                                             v
              independent gates (validation landmarks never fitted)
                                             v
          keyframe CALIBRATED / UNCALIBRATED / INVALIDATED (+ reason)
                                             v
     temporal policy: shots, declared camera motion, suspected cuts
                                             v
     every analysed frame: CALIBRATED (exact or verified propagation)
                           UNCALIBRATED | INVALIDATED  → no matrix
                                             v
AI2C player-observation/v1 ──> projected-player-observation/v1
                               pitch_xy_m only on CALIBRATED frames
```

## Contracts

| Contract | Purpose |
|---|---|
| `footballai.pitch-landmark-annotations/v1` | Manual input: source checksums and native frame size, explicit pitch geometry, shots, and per frame the clicked landmarks with role, quality, review status and annotator |
| `footballai.pitch-calibration/v1` | Output: pipeline and software versions, source and annotation checksums, pitch and coordinate system, policy, shots, suspected cuts, keyframes (correspondences, residuals, metrics, checks, homography if valid), propagation intervals, a state for every analysed frame, summary, limitations, integrity checksum |
| `footballai.projected-player-observation/v1` | An unchanged AI2C `player-observation/v1` record plus `pitch_xy_m`, projection status and calibration provenance |
| `footballai.pitch-calibration-benchmark/v1` | Synthetic offline benchmark |

### Integrity of `footballai.pitch-calibration/v1`

- It is deterministic JSON with sorted keys and no wall-clock fields. The same
  inputs in the same software environment give the same bytes.
- It stores every input needed to recompute itself. Loading the file, or
  constructing it in memory, **re-derives** every keyframe homography,
  interval, frame state and the summary, and rejects any difference.
  - An edited matrix fails.
  - An `uncalibrated` frame relabelled `calibrated` fails.
  - A frame carrying another frame's matrix fails.
  - A changed landmark fails.
  - A changed policy fails when it changes any outcome.
  - A deleted suspected cut fails.
- `integrity.content_sha256` catches accidental corruption. Consumers also pin
  the file's own SHA-256.
- The trust anchor is the annotation evidence, pinned by
  `source.annotations_sha256`. Forging coordinates requires forging landmark
  clicks that independently pass validation.

## Pitch geometry and coordinate system

`footballai.pitch-plan/v1` is the plan view seen from above. LEFT, RIGHT, TOP
and BOTTOM name sides of the plan drawing, never sides of the image.

- **Origin:** where the LEFT goal line meets the TOP touchline.
- **`x_m`:** runs along the touchlines, from the LEFT goal line (0) to the
  RIGHT goal line (`length_m`).
- **`y_m`:** runs along the goal lines, from the TOP touchline (0) to the
  BOTTOM touchline (`width_m`).
- **Handedness:** image-like (x right, y down). A camera above the pitch sees
  its upper surface, so a valid image→pitch homography always has a positive
  Jacobian. A negative Jacobian is a mirrored annotation.

`PitchGeometry` has **no default size**. Length, width, markings and a
`dimensions_source` string are required, and length must exceed width. Law 1
markings are available as the named constant `IFAB_LAW_1_MARKINGS`
(penalty area 16.5 × 40.32 m, goal area 5.5 × 18.32 m, penalty mark 11 m,
centre circle and penalty arc 9.15 m). `within_law_1_range` is recorded but
not enforced, because youth pitches are legitimate.

There are 31 landmarks, all line intersections or marks with an exact
definition:
- 4 corners
- 2 halfway-line ends, the centre mark, and the 2 points where the halfway
  line meets the centre circle
- for each side:
  - 4 penalty-area points and 4 goal-area points: two on the goal line, two
    front corners
  - the penalty mark
  - 2 points where the penalty arc meets the penalty-area front line

The `landmarks` CLI command prints them and draws a labelled plan SVG.

Each shot records `camera_side` (`bottom_touchline`, `top_touchline`,
`left_goal_line`, `right_goal_line`, `unknown`). For a camera without roll,
moving down the image moves toward the camera on the pitch. A homography whose
direction disagrees with the declared side is invalidated. `unknown` skips
this check and never guesses.

## Annotation workflow

Pixel coordinates follow the AI1 box convention: continuous, origin at the
top-left corner of the frame, native resolution. Frame indices are AI1 source
frame indices.

```bash
M="python -m footballai_v2.execution.ai3.cli"
PITCH="--length-m 105 --width-m 68 --markings ifab-law-1 --dimensions-source 'ground survey'"
```

1. Print the landmark table and draw a labelled plan:

   ```bash
   $M landmarks $PITCH --svg /tmp/plan.svg
   ```

2. Create an empty annotation file bound to the AI1 checksums:

   ```bash
   $M template $PITCH --source-id clip --detections det.parquet \
     --tracklets trk.parquet --shot s1:0:449:static:bottom_touchline \
     --keyframe 9 --keyframe 239 --output annotations.json
   ```

3. Extract one frame and write a local click page:

   ```bash
   $M annotate-frame $PITCH --video clip.mp4 --frame-index 9 \
     --timestamp 1.8 --shot-id s1 --output-dir ~/footballai-frames
   ```

   This command refuses to write inside a Git working tree. Open the HTML
   page, click each visible landmark, tick "validation" for roughly one in
   three, and export. Paste the exported frame into `annotations.json`.
   Exported points start as `pending`.

4. A reviewer sets each point's `review_status` to `accepted` or `rejected`.
   Only `accepted` points are used.

5. Check the file and dry-run each keyframe:

   ```bash
   $M validate --annotations annotations.json --tracklets trk.parquet \
     --detections det.parquet
   ```

6. Calibrate:

   ```bash
   $M calibrate --annotations annotations.json --tracklets trk.parquet \
     --detections det.parquet --suspected-cuts-from-tracklets \
     --output calibration.json
   ```

The reader rejects any of the following:
- unknown landmarks
- duplicate landmarks
- pitch coordinates that disagree with the configured geometry (for example,
  annotations made for another pitch size)
- pixels outside the frame
- two landmarks within 1 px of each other
- a frame outside its shot
- overlapping shots
- unknown fields
- non-finite numbers

`calibrate` and `validate` additionally reject:
- a tracklets, detections or video checksum that differs from the one the
  annotations declare
- a frame size that differs from the source's
- an annotated frame whose timestamp disagrees with the AI1 frame

## Estimator

The estimator maps image pixels to pitch metres with a normalised DLT (Hartley
normalisation, SVD in NumPy) inside a deterministic RANSAC.
- **Sampling:** every 4-point subset is enumerated in a fixed order when there
  are at most 5,000 of them (always true for hand annotation). Larger sets use
  sampling from `random.Random(seed + frame_index)`, which is stable across
  Python and NumPy versions.
- **Degenerate samples:** a minimal sample with three nearly collinear points,
  in either space, is skipped.
- **Scoring:** most inliers, then the lowest truncated squared error, then the
  earliest subset.
- **Refinement:** the inliers are refitted until the inlier set is stable.
- **Error metric:** every error is a geometric distance on the pitch, in
  metres.
- **Determinism:** each matrix is scaled to unit Frobenius norm with a
  positive denominator at the fit centroid, then rounded to 12 significant
  digits. All metrics come from the rounded matrix, so a stored artifact can
  be re-verified.
- **OpenCV cross-check:** the estimator agrees with `cv2.findHomography` on
  noisy data (within 5 cm). The test is skipped where OpenCV is not installed,
  as in CI.

OpenCV is used only by `annotate-frame`, to read one frame.

## Quality gates

`CalibrationPolicy` (`ai3-calibration-policy/v1`) is versioned, validated, and
recorded in every artifact. The defaults were fixed **before** any benchmark
ran, from the arguments below. They were not tuned on synthetic or real
results.

| Parameter | Default | Rationale |
|---|---|---|
| `min_fit_points` / `min_fit_inliers` | 4 / 4 | Mathematical minimum (8 degrees of freedom). Catching an outlier needs at least 5 |
| `min_inlier_ratio` | 0.6 | If over 40% of a person's clicks disagree with the consensus, the frame's annotation is unreliable |
| `min_validation_points` | 2 | Smallest held-out sample with both a median and a maximum. **Weak**; see results |
| `ransac_threshold_m` | 1.0 m | Neighbouring landmarks are at least 5.5 m apart, so a mislabel errs by 5.5 m or more. Far-field click error is about 2–3 px × 0.1–0.3 m/px |
| `max_validation_median_error_m` / `max_validation_error_m` | 0.75 / 1.5 m | Coach zones are at least ~10 m wide, and box foot-point bias is ~0.5 m. Calibration error should stay at or below that order |
| `min_image_hull_fraction` | 0.05 | Fit landmarks must span at least 5% of the frame. A homography extrapolates poorly |
| `min/max_metres_per_pixel` | 0.002 / 1.0 | Sanity range from extreme close-up to a whole pitch in about 100 px |
| `min_triangle_ratio` | 0.02 | Collinearity test for minimal samples |
| `min_dlt_singular_ratio` | 1e-9 | Numerical rank deficiency |
| `accepted_review_statuses` | `accepted` | Unreviewed clicks never calibrate |
| `validation_requires_clear` | true | Estimated (occluded) points are not used as validation |
| `propagate_static_shots` | true | See the temporal policy |

A keyframe is `uncalibrated` when there is too little evidence:
- `insufficient_fit_points`
- `insufficient_validation_points`
- `degenerate_or_collinear_geometry`
- `insufficient_ransac_inliers`
- `ill_conditioned_estimate`

It is `invalidated` when a homography was estimated but a check rejects it.
The checks run in this order:
1. `non_finite_or_singular_homography`
2. `landmarks_beyond_camera_horizon`
3. `reflected_orientation`
4. `camera_side_inconsistent`
5. `implausible_metres_per_pixel`
6. `insufficient_spatial_coverage`
7. `validation_error_exceeds_threshold`

Each keyframe reports:
- fit, validation and excluded point counts
- RANSAC inliers and outliers, with per-point flags
- fit RMSE and maximum error, and the residual degrees of freedom (a fit of
  exactly 4 points has residual 0, which proves nothing)
- held-out median, maximum and p90 error (p90 only when there are 10 or more
  validation points)
- validation points outside the fit hull
- image-hull fraction and pitch-hull area
- metres per pixel
- RANSAC subsets evaluated and degenerate
- every check's pass, fail or not-checked result

Fit residual is never reported as accuracy. Reprojection error is never
reported as tracking accuracy.

## Camera cuts and moving cameras

- Shots are explicit, manual, non-overlapping frame intervals with a declared
  `camera_motion` (`static`, `moving` or `unknown`). Frames outside every shot
  are `uncalibrated` (`no_shot_segment`).
- A keyframe's homography is valid for that exact frame.
- Other frames in a shot declared `static` may use a keyframe's homography
  only if all of these hold:
  - the frame lies between two consecutive keyframes that are both
    `calibrated`
  - each of the two homographies reprojects the other's inlier and validation
    landmarks within `max_validation_error_m`
  - no suspected cut falls between them

  The nearest keyframe's homography is then used, and the frame records
  `method: static_shot_keyframe_propagation` and `source_keyframe`.
- If the two keyframes disagree, the frames between them are `invalidated`
  (`static_shot_keyframes_disagree`). If a suspected cut falls between them,
  those frames are `invalidated` (`suspected_camera_cut`).
- Frames before the first or after the last keyframe of a static shot are
  `uncalibrated`.
- `moving` and `unknown` shots never propagate (`camera_not_declared_static`).
- Suspected cuts may come from the AI2C birth-burst rule (8 or more tracklet
  births in one frame, `--suspected-cuts-from-tracklets`). The rule is
  unevaluated, so it can only remove calibration, never add it. AI3 does not
  claim automatic shot detection.

A camera pan therefore produces no pitch movement. In the tests, a stationary
player crosses over 100 px of the image during an annotated pan and receives
exactly one pitch position, on the keyframe.

## Projection of AI2C observations

`project_player_observations(observations, calibration, calibration_sha256)`
yields one `ProjectedPlayerObservation` per AI2C `PlayerObservation`, in input
order.

**Refusals.** The call raises an error when:
- the calibration checksum is wrong
- the calibration is not bound to a tracklets artifact
- the observation's tracklets artifact is a different one
- a timestamp disagrees, which means the frame conventions differ
- the calibration was gated by a policy other than the expected one
  (`expected_policy`, the default policy unless one is passed)
- the calibration is synthetic and `allow_synthetic=True` was not passed

**Per-observation status.** Otherwise each output carries `pitch_xy_m=None`
and a status of `frame_not_in_calibration`, `frame_uncalibrated`,
`frame_invalidated`, `foot_point_outside_image`,
`foot_point_beyond_camera_horizon`, `non_finite_projection` or
`outside_pitch_margin` (more than 5 m beyond the lines). Only on a `calibrated`
frame does it carry `projected` with `(x_m, y_m)`, plus `inside_pitch_lines`
and `extrapolated_beyond_landmarks` (the foot point lies outside the keyframe's
fit-landmark hull).

**Guarantees:**
- The AI2C record is embedded unchanged. Its own guard, which rejects pitch
  coordinates, is untouched.
- Pixel boxes, foot pixel, identity status, provisional or unknown identity,
  and provenance checksums pass through. The calibration checksum is added.
- `ProjectedPlayerObservation` recomputes its projection from the embedded
  foot pixel and homography, so a caller cannot build one with injected
  coordinates.
- Nothing is aggregated. Projection never upgrades identity: AI2C still
  yields only `confirmed` identities by default.

## Results

### Synthetic benchmark

Source: `evaluation/ai3/synthetic-benchmark.json`, generated with
`python -m footballai_v2.execution.ai3.cli benchmark`.

**Setup:**
- **Scenes:** pinhole cameras behind the BOTTOM touchline, 20–45 m back,
  12–28 m high, focal length 900–2600 px, random pan, 1920×1080 frames, an
  explicitly configured 105 × 68 m pitch.
- **Validation points:** one in three visible landmarks.
- **Partitions:** 40 development seeds were used while building the pipeline.
  200 evaluation seeds were scored once at the end, with thresholds unchanged.
- **Ground truth:** the true camera is known, so the benchmark also measures
  the true error over a 2 m grid of the visible pitch.
- **Unsafe acceptance:** a calibrated frame whose true p95 field error exceeds
  2 m. This threshold was declared in advance.

Evaluation partition (200 scenes per scenario):

| Scenario | Calibrated | Uncal. | Invalid. | True p95 field error (median / p90 / max) | Unsafe acceptances |
|---|---|---|---|---|---|
| clean, 1 px clicks | 180 (90.0%) | 7 | 13 | 0.28 / 0.77 / 2.19 m | **1 / 180** |
| noisy, 3 px clicks | 90 (45.0%) | 11 | 99 | 0.73 / 2.00 / 5.95 m | **8 / 90** |
| one fit outlier, 1 px | 163 (81.5%) | 8 | 29 | 0.30 / 0.71 / 2.57 m | **3 / 163** |

| Invalid input | Rejected (not calibrated) |
|---|---|
| LEFT↔RIGHT mirrored labels | **200 / 200** (191 reflected, the rest too few points) |
| Pitch filmed at 100 × 64 m, configured as 105 × 68 m | **106 / 200 (53%)** |

The development partition (40 scenes per scenario) is consistent:
- clean: 0 unsafe in 37 calibrated
- noisy: 1 unsafe in 20 calibrated
- one fit outlier: 0 unsafe in 32 calibrated
- mirrored: 40 / 40 rejected
- wrong pitch size: 21 / 40 rejected

The temporal scenario has 210 frames: four shots and a gap, with a genuine
pan, a false "static" claim and a suspected cut. Every expected frame state
was observed. Of 900 stationary-player observations, 264 were projected
(true error median 0.06 m, max 0.26 m), 402 fell on uncalibrated frames and
234 on invalidated frames. During the un-keyframed pan, 0 were projected.

Runtime: about 50 ms per keyframe without tracing. 1,207 keyframes took 315 s
under `tracemalloc`, with a peak traced memory of 2.6 MB. No inference was
run.

### What the results mean

- **The gates catch structural mistakes.** Mirrored labels, collinear layouts,
  too few points and wrong declared camera sides are rejected, and
  uncalibrated frames never get coordinates.
- **A small held-out set cannot bound extrapolation.** With 2–6 validation
  landmarks, a calibration can pass every check while its error far from the
  landmarks exceeds 2 m: 1 in 180 clean frames, 8 in 90 at 3 px. Projections
  outside the fit hull are therefore flagged `extrapolated_beyond_landmarks`.
  A real benchmark must decide whether more validation points, a hull-coverage
  requirement for validation points, or stricter thresholds are needed.
  Thresholds were deliberately **not** changed after seeing these numbers.
- **A wrong pitch size is not reliably detectable.** Landmarks inside one half
  (penalty and goal areas) have fixed Law 1 sizes, so they fit any pitch
  length. Correct pitch dimensions must be measured and configured; the gates
  cannot catch this reliably.

### Real footage

**Pending.** There are 0 independently annotated real frames. The two licensed
local clips used by AI2A and AI2B are referenced by checksum only and are not
in this repository or session.

### Not measured

- real broadcast accuracy
- human click error and inter-annotator agreement
- lens distortion, pitch non-planarity, line blur and occlusion
- foot-point bias of box bottoms
- quality of the birth-burst cut rule
- cross-camera and cross-stadium generalisation

## What AI3 unlocks, and what it does not

AI3 provides a verified way to put AI1 tracklet observations on the pitch,
frame by frame, with coverage and quality states. These downstream features
can now be built on it:
- team occupancy maps and team heatmaps
- team width, length and compactness
- pitch-zone filters
- position-aware identity gating

None of these is implemented in AI3. Each must carry units, coverage,
uncertainty (including the extrapolation flag), quality status, time interval
and artifact hashes.

Individual heatmaps, displacement, distance and speed also need reviewed
identities (AI2C `confirmed`) and validated calibration over the relevant
frames.

The copilot, its five read-only tools, labels and D7 rules are unchanged. AI3
is not wired into the `pitch_calibration` run stage yet.

## Next milestone

1. Annotate a small real benchmark with the included tooling:
   - frames from at least two camera sequences of the licensed clips, pinned
     by video checksum, with pixels kept outside Git
   - at least 8–10 validation points per frame, spread beyond the fit hull
   - a second annotator on a subset, to measure click agreement

   Report held-out error and coverage per sequence.
2. Then decide the validation-point requirements from that evidence.
3. Then build team-level pitch occupancy maps over `projected` observations,
   with explicit coverage.
