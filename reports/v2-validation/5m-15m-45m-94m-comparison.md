# V2 `v1_compat` Full-Match Acceptance — 5 / 15 / 45 / 94 Minute Comparison

_Generated 2026-07-28T16:49:55.920305Z. Sources referenced by label only; run artifacts live under the gitignored run root and are not committed. The historical three-run report (`5m-15m-45m-*`) is preserved unchanged._

## Acceptance decision: **PASS WITH LIMITATIONS**

The 94-minute full-match run completed end-to-end through the real application workflow (FastAPI streaming upload → filesystem queue → separate worker → `v1_compat`) on Apple MPS with the **same effective configuration** as the validated 5/15/45-minute runs, and passed full artifact-integrity verification. It is rated **PASS WITH LIMITATIONS** because tracking fragmentation is severe (below), which caps analytical fidelity even though the pipeline is functionally and operationally sound.

## Executive summary

- **Functional acceptance — PASS.** Upload → queue → worker → all required stages succeeded; unsupported stages correctly skipped; 7/7 artifacts checksum-verified; input SHA-256 matches the source; device stayed **MPS** with no CPU fallback and no worker restart.
- **Performance acceptance — PASS with a caveat.** Total processing **16,084.4 s (~4.47 h)** for a 95.654-minute video → **2.803× real-time** (168.15 s per video-minute). This is **markedly slower than the 45-minute run** (1.375× real-time): effective throughput roughly halved (3.636 → 1.784 frames/s). The machine was held awake (`caffeinate`) for the whole run, so the slowdown is genuine sustained-load degradation — most plausibly thermal throttling over ~4.5 h of continuous MPS inference compounded by growing tracker state — not a sleep artifact. Budget generously for the 94-minute class.
- **Analytical acceptance — LIMITED.** Detection stays consistent (~5.365–6.84 det/frame), but **97.65% of raw tracks are too short to score** and raw ByteTrack IDs reach 8,099 for a ~22-player match — heavy fragmentation with no re-ID and no homography.
- **Temporal overlap verified.** The 94-minute video shares the same opening first half as the 45-minute run (frame-hash Hamming ≤ 4/256 across six timestamps to 2400 s).
- **Source note.** The full-match source is **1920×1080** (the 5/15/45 clips were 1280×720); inference still runs at `image_size=1280`, so detector configuration parity is preserved.

## Run identification

| Video | Run ID | Logical analysis ID | Status | Duration (s) | Input size | Resolution | SHA-256 prefix | Created (UTC) |
|---|---|---|---|---|---|---|---|---|
| 5-minute source | `0593bf4d-942a-4d0d-a577-f38ebb9e32d1` | `928cd088-7aa7-46c2-bc85-50d6b1ea3f93` | succeeded | 275.833 | 316.6 MB | 1280x720 | `13fca791280a` | 2026-07-27T19:12:50.025667Z |
| 15-minute source | `517161f6-ca9d-496e-aab3-6f72de3e6fd2` | `ba993347-6a98-4cbd-ac10-d4ae45174aa9` | succeeded | 918.628 | 1,054.4 MB | 1280x720 | `22783fd6bf2c` | 2026-07-27T20:42:11.633849Z |
| 45-minute source | `aca8de80-2fb2-4ce4-953a-5af33259ac48` | `5617755c-b0ab-4442-ae48-ee43c14e6d25` | succeeded | 2,772.579 | 3,180.4 MB | 1280x720 | `5ee5b9ba945b` | 2026-07-27T21:24:57.175822Z |
| 94-minute full-match source | `2e8d5dee-5c3e-4080-bc6d-e4a39db35983` | `9d3a0d5c-a0ea-4ded-b2ff-7fd16c2f71f2` | succeeded | 5,739.253 | 8,305.3 MB | 1920x1080 | `fcf5418f08a6` | 2026-07-28T12:19:27.095388Z |

The 94-minute run was matched to its source by **exact input SHA-256** and is a newly executed attempt (no prior run existed for this checksum).

## Configuration parity

| Parameter | 5-minute source | 15-minute source | 45-minute source | 94-minute full-match source |
|---|---|---|---|---|
| Pipeline version | v1_compat/1.0.0 | v1_compat/1.0.0 | v1_compat/1.0.0 | v1_compat/1.0.0 |
| Model | yolov8m.pt | yolov8m.pt | yolov8m.pt | yolov8m.pt |
| Model SHA-256 prefix | 5d4a90cdc7a2 | 5d4a90cdc7a2 | 5d4a90cdc7a2 | 5d4a90cdc7a2 |
| Selected device | mps | mps | mps | mps |
| Requested device | auto | auto | auto | auto |
| Target FPS | 5 | 5 | 5 | 5 |
| Image size | 1,280 | 1,280 | 1,280 | 1,280 |
| Confidence | 0.2 | 0.2 | 0.2 | 0.2 |
| m per px (scale const.) | 0.086 | 0.086 | 0.086 | 0.086 |

Detector configuration is **identical** across all four runs. Only the source resolution differs (1080p vs 720p), which does not change inference settings.

## Performance comparison

| Metric | 5-minute source | 15-minute source | 45-minute source | 94-minute full-match source |
|---|---|---|---|---|
| Input duration (s) | 275.833 | 918.628 | 2,772.579 | 5,739.253 |
| Video minutes | 4.597 | 15.31 | 46.21 | 95.654 |
| Queue wait (s) | 0.17 | 0.138 | 0.2 | 0.287 |
| Total processing (s) | 301.68 | 865.18 | 3,812.51 | 16,084.4 |
| Processing per video-min (s) | 65.62 | 56.51 | 82.5 | 168.15 |
| Processing / video ratio | 1.094 | 0.942 | 1.375 | 2.803 |
| Frames processed | 1,379 | 4,592 | 13,862 | 28,696 |
| Effective FPS over processing | 4.571 | 5.308 | 3.636 | 1.784 |

### Scalability classification

Processing-time scaling factors vs. video-length factors:

- **5→15:** 2.87× time for 3.33× video — sub-linear.
- **15→45:** 4.41× time for 3.02× video — super-linear.
- **45→94:** 4.22× time for 2.07× video — super-linear.
- **5→94:** 53.32× time for 20.81× video.

Per-video-minute cost across the four runs: 65.62 → 56.51 → 82.5 → 168.15 s/video-min. Overall the behavior is **inconsistent**: efficient at short lengths, then super-linear at longer durations — extrapolate to future runs assuming the per-minute cost can rise, not hold.

## Detection and tracking comparison

| Metric | 5-minute source | 15-minute source | 45-minute source | 94-minute full-match source |
|---|---|---|---|---|
| Total detections | 7,399 | 24,919 | 83,941 | 196,272 |
| Detections / video-min | 1,609.452 | 1,627.579 | 1,816.525 | 2,051.891 |
| Detections / processed frame | 5.365 | 5.427 | 6.055 | 6.84 |
| Raw ByteTrack tracks | 401 | 1,365 | 4,406 | 8,099 |
| Raw tracks / video-min | 87.227 | 89.155 | 95.348 | 84.67 |
| Scored tracks (summary roster) | 23 | 22 | 70 | 190 |
| Advisory-scored tracks (coverage-gated) | 23 | 22 | 24 | 23 |
| Insufficient tracks | 378 | 1,343 | 4,336 | 7,909 |
| Insufficient-track % | 94.26 | 98.39 | 98.41 | 97.65 |
| Fragmentation (raw / scored) | 17.43 | 62.05 | 62.94 | 42.63 |
| Max observations on one track | 64 | 126 | 95 | 1,735 |
| Mean obs / raw track | 18.45 | 18.26 | 19.05 | 24.23 |

Detection density stays in a narrow band across all four durations (5.365–6.84 det/frame), so the detector behaves consistently at full-match length. **Note a confound:** the 94-minute source is 1080p while the others are 720p, which likely explains why its detections-per-frame sits at the top of the band — so the detection-density comparison is indicative, not perfectly controlled. **Tracking fragments badly:** raw IDs climb 401 → 1,365 → 4,406 → 8,099 while the real roster is ~22, and the insufficient-track share sits at ~94–98%. **More tracks here is more fragmentation, not better tracking** — without re-identification and with an uncompensated moving broadcast camera, single players are repeatedly split into many short IDs. One nuance at full length: the longest track persisted for 1,735 observations (vs 95 at 45 min) and the summary roster grew to 190, so the raw/scored fragmentation *ratio* actually eased (62.94× → 42.63×) even as absolute fragmentation grew — a few players track well for long stretches while the majority still fragment. Tracker-state growth (thousands of candidate IDs to manage) is also a likely contributor to the super-linear processing cost.

## Artifact and storage comparison

| Metric | 5-minute source | 15-minute source | 45-minute source | 94-minute full-match source |
|---|---|---|---|---|
| Artifact count | 7 | 7 | 7 | 7 |
| Artifact total (bytes) | 210,924 | 205,651 | 619,317 | 1,637,699 |
| Artifact MB / video-min | 0.046 | 0.013 | 0.013 | 0.017 |
| Input MB / video-min | 68.87 | 68.87 | 68.83 | 86.83 |

Artifacts remain small (hundreds of KB) and grow far slower than the video, because they summarize a bounded roster rather than per-frame data. Run-directory storage is dominated by the copied source video — the 94-minute run directory is ~8.3 GB, almost entirely the input copy.

## Common-window comparison (45 vs 94 minutes)

Temporal overlap was **verified** by bounded FFmpeg frame sampling of the 45-minute stored input against the 94-minute source (single frames, 64×64 grayscale, 16×16 average-hash, Hamming distance out of 256):

| Timestamp (s) | Hamming (45m vs 94m) / 256 |
|---|---|
| 1 | 3 |
| 30 | 2 |
| 120 | 0 |
| 240 | 1 |
| 900 | 4 |
| 2400 | 0 |

Distances of 0–4/256 (despite the 720p↔1080p difference) confirm the 94-minute video is the **same match from the same kickoff**; its opening ~40 minutes coincide with the 45-minute clip. We therefore compare aggregate/spatial behavior over the shared opening — **not** ByteTrack IDs (independent per run) or per-player totals (V1 selects top-N tracks over each clip's whole horizon, so rosters differ). The 94-minute second half has no reference clip to compare against.

## Data-quality limitations (scientific limits)

- **Unverified identities** — no shirt-number or re-identification; scored "players" are the longest surviving tracks, not confirmed individuals.
- **ID switches remain likely** — 97.65% of raw tracks in the 94-minute run are too short to score.
- **No re-identification** and **no homography calibration** — positions use a single constant metres-per-pixel; distances/speeds are approximate and not perspective-correct.
- **Camera motion is uncompensated**, inflating apparent motion and breaking tracks.
- **Movement values are approximate** heuristics; treat cross-duration magnitudes as indicative only.
- **Workload & Fatigue Advisory is heuristic and advisory only — not diagnosis or clinical advice.** Second-half `distance_drop` remains a degenerate artifact and must not be compared as medically meaningful.

## Recommendation / next precautions

The full-match acceptance test is **PASS WITH LIMITATIONS**: the workflow ingests, queues, processes, and publishes a full 94-minute match on MPS with verified integrity and configuration parity. The limitations are (a) **analytical** — severe tracking fragmentation — and (b) **operational throughput** — sustained-load slowdown to ~2.8× real-time. Neither blocks acceptance; both shape how the system should be used.

Precautions for routine full-match runs:

- **Time budget ~2.8–3× real-time (~4.5+ h for a 94-min match)**, not the ~1.4× the 45-min run suggested — plan for throughput to fall as the run lengthens (thermal/tracker-state). Keep a keep-awake (`caffeinate`) active for the whole run.
- Keep bounded upload/duration limits just above the real file (this run used a 9 GiB upload cap and a 7200 s duration cap for a 7.7 GiB / 5,739 s source), ensure free disk ≈ 2× the source, keep the device on **MPS**, and never launch a second heavy analysis concurrently.
- Before treating per-player analytics as authoritative, prioritize a **re-identification + pitch-homography** upgrade and camera-motion compensation to attack the fragmentation root cause.

