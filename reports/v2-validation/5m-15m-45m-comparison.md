# V2 `v1_compat` Analysis — 5 / 15 / 45 Minute Comparison

_Generated 2026-07-27T22:31:21.313041Z. Source videos are referenced by label only; run artifacts live under the gitignored run root and are not committed._

## Executive summary

- **Scales successfully but not for free.** Three real `v1_compat` runs (5 / 15 / 45 min) of the same Belgium–Japan first half completed end-to-end on Apple MPS with identical detector configuration and **zero errors, zero worker restarts, and full artifact-integrity verification** (7/7 artifacts checksum-verified per run).
- **Processing is super-linear at longer durations.** Wall-clock per video minute was **65.62 → 56.51 → 82.5 s/video-min** (processing-to-video ratio 1.094× → 0.942× → 1.375×). It stayed near real-time through 15 min but **rose ~46% per minute from 15→45 min** — the 15→45 step cost 4.41× the time for only 3.02× the video. Effective throughput fell 5.308 → 3.636 frames/s.
- **Detection density is consistent.** Detections per processed frame stay in a narrow band (~5.365–6.055); total detections grow roughly in proportion to duration, so the detector itself behaves consistently.
- **Major quality limitation — tracking fragmentation.** Raw ByteTrack identities per scored player rise steeply with duration (17.43× → 62.05× → 62.94×), so 98.41% of raw tracks are too short to score in the 45-minute run. This is expected for V1 (no re-identification, no homography, moving broadcast camera) but it caps analytical fidelity.
- **Bottleneck.** The detection/tracking stage dominates wall-clock (>99%); metrics, advisory and publication are sub-second. Any speed-up work belongs in detection/tracking, and the super-linear component is the tracker managing an ever-growing set of fragmented IDs.
- **94-minute acceptance test: ready with precautions** (see Recommendation).

## Run identification

| Video | Run ID | Logical analysis ID | Status | Duration (s) | Input size | SHA-256 prefix | Created (UTC) |
|---|---|---|---|---|---|---|---|
| 5-minute source | `0593bf4d-942a-4d0d-a577-f38ebb9e32d1` | `928cd088-7aa7-46c2-bc85-50d6b1ea3f93` | succeeded | 275.833 | 316.6 MB | `13fca791280a` | 2026-07-27T19:12:50.025667Z |
| 15-minute source | `517161f6-ca9d-496e-aab3-6f72de3e6fd2` | `ba993347-6a98-4cbd-ac10-d4ae45174aa9` | succeeded | 918.628 | 1,054.4 MB | `22783fd6bf2c` | 2026-07-27T20:42:11.633849Z |
| 45-minute source | `aca8de80-2fb2-4ce4-953a-5af33259ac48` | `5617755c-b0ab-4442-ae48-ee43c14e6d25` | succeeded | 2,772.579 | 3,180.4 MB | `5ee5b9ba945b` | 2026-07-27T21:24:57.175822Z |

All three runs matched their source video by **exact input SHA-256** (strongest identifier); byte size, duration, and provenance filename agree.

## Configuration parity

| Parameter | 5-minute source | 15-minute source | 45-minute source |
|---|---|---|---|
| Pipeline version | v1_compat/1.0.0 | v1_compat/1.0.0 | v1_compat/1.0.0 |
| Model | yolov8m.pt | yolov8m.pt | yolov8m.pt |
| Model SHA-256 prefix | 5d4a90cdc7a2 | 5d4a90cdc7a2 | 5d4a90cdc7a2 |
| Selected device | mps | mps | mps |
| Requested device | auto | auto | auto |
| Target FPS | 5 | 5 | 5 |
| Image size | 1,280 | 1,280 | 1,280 |
| Confidence | 0.2 | 0.2 | 0.2 |
| m per px (scale const.) | 0.086 | 0.086 | 0.086 |
| Sprint threshold (m/s) | 5.5 | 5.5 | 5.5 |

Configuration is **identical** across all three runs — the comparison is configuration-matched, not configuration-mismatched.

## Performance comparison

| Metric | 5-minute source | 15-minute source | 45-minute source |
|---|---|---|---|
| Input duration (s) | 275.833 | 918.628 | 2,772.579 |
| Video minutes | 4.597 | 15.31 | 46.21 |
| Queue wait (s) | 0.17 | 0.138 | 0.2 |
| Total processing (s) | 301.68 | 865.18 | 3,812.51 |
| Processing per video-min (s) | 65.62 | 56.51 | 82.5 |
| Processing / video ratio | 1.094 | 0.942 | 1.375 |
| Frames processed | 1,379 | 4,592 | 13,862 |
| Effective FPS over processing | 4.571 | 5.308 | 3.636 |

**Scaling factors (processing time vs. video length):** 5→15 min = 2.87× time for 3.33× video (**sub-linear** — the tracker is efficient at short lengths); 15→45 min = 4.41× time for 3.02× video (**super-linear** — clearly slower per minute); 5→45 min = 12.64× time for 10.05× video (**super-linear** overall). Net: processing cost per video-minute is not flat — it dips then climbs, so wall-clock grows **faster than linearly** as clips lengthen. Extrapolating to 94 min should assume the per-minute cost keeps rising, not that it holds at the 45-min rate.

## Detection and tracking comparison

| Metric | 5-minute source | 15-minute source | 45-minute source |
|---|---|---|---|
| Total detections | 7,399 | 24,919 | 83,941 |
| Detections / video-min | 1,609.452 | 1,627.579 | 1,816.525 |
| Detections / processed frame | 5.365 | 5.427 | 6.055 |
| Raw ByteTrack tracks | 401 | 1,365 | 4,406 |
| Raw tracks / video-min | 87.227 | 89.155 | 95.348 |
| Scored tracks (summary roster) | 23 | 22 | 70 |
| Advisory-scored tracks (coverage-gated) | 23 | 22 | 24 |
| Insufficient tracks | 378 | 1,343 | 4,336 |
| Insufficient-track % | 94.26 | 98.39 | 98.41 |
| Fragmentation (raw / scored) | 17.43 | 62.05 | 62.94 |
| Max observations on one track | 64 | 126 | 95 |
| Mean obs / raw track | 18.45 | 18.26 | 19.05 |

**Scale vs. fragmentation.** Detection scales cleanly with duration and detections-per-frame stays in a narrow band, so the detector behaves consistently on the longer clips. **Track counts, however, grow far faster than the number of real players.** The summary roster is 23 / 22 / 70 scored tracks while raw ByteTrack IDs climb 401 → 1,365 → 4,406. More tracks here means *more fragmentation*, not better tracking: without re-identification and with a moving broadcast camera, ByteTrack repeatedly breaks a single player into many short-lived IDs, so ~94–98% of raw tracks never reach scoring length. Note two different "scored" counts: the track/team summary keeps every track above a minimum length (70 for 45 min), but the workload advisory further gates on coverage and scores far fewer (24 for 45 min) — a sign that most "scored" tracks are still too fragmentary for even heuristic fatigue output at full length.

## Artifact and storage comparison

| Metric | 5-minute source | 15-minute source | 45-minute source |
|---|---|---|---|
| Artifact count | 7 | 7 | 7 |
| Artifact total (bytes) | 210,924 | 205,651 | 619,317 |
| Artifact MB / video-min | 0.046 | 0.013 | 0.013 |
| Input MB / video-min | 68.87 | 68.87 | 68.83 |

Published artifact payloads stay **small** — hundreds of KB — and grow far slower than the video (210,924 → 205,651 → 619,317 bytes), because they summarize a bounded scored roster rather than per-frame data. Per-minute artifact size is flat-to-falling. Storage growth is dominated **entirely by the copied source video** (a constant ~68.87 MB per video-minute, so the 45-min run's run directory is ~3 GB — almost all of it the input copy).

## Common-window comparison

Temporal overlap was **verified** by bounded FFmpeg frame sampling (single frames at t = 1, 30, 120, 240 s, downscaled to 64×64 grayscale, 16×16 average-hash, Hamming distance out of 256 bits):

| Timestamp | 5m vs 15m | 5m vs 45m | 15m vs 45m |
|---|---|---|---|
| 1 s | 0 | 0 | 0 |
| 30 s | 1 | 3 | 2 |
| 120 s | 0 | 0 | 0 |
| 240 s | 0 | 0 | 0 |

Frames at t = 1/120/240 s are pixel-identical across all three videos (Hamming 0/256); at t = 30 s they differ by only 1–3 bits (re-encoding noise). The downscaled t = 1 s frame is byte-identical for the 15- and 45-minute clips. **Conclusion: all three clips are the same match from the same kickoff** — the 5-minute clip is the opening ~4.6 min, the 15-minute clip the opening ~15.3 min, and the 45-minute clip the full ~46.2 min first half.

**What this does and does not license.** Because the frames overlap, detection behavior on the shared opening window is directly comparable, and detections-per-frame being near-constant (~5.4) across runs is consistent with the same underlying content. It does **not** license per-track equality: (a) ByteTrack IDs are independent between runs, and (b) V1 selects the top-N longest tracks over the *whole* clip, so each run's scored roster is chosen over a different horizon. We therefore compare spatial/aggregate detection behavior, not track IDs or per-player totals, across the common window.

## Data-quality limitations

- **Unverified identities.** Track identities are not verified; there is no shirt-number or re-identification step. Scored "players" are the longest surviving tracks, not confirmed individuals.
- **Probable ID switches / fragmentation.** 98.41% of raw tracks in the 45-min run are too short to score — strong evidence of frequent ID switches under camera motion.
- **No homography calibration.** Positions use a single constant metres-per-pixel (0.086); there is no pitch homography, so distances/speeds are approximate and not perspective-correct.
- **Camera-motion impact.** A moving broadcast camera is not compensated, inflating apparent motion and breaking tracks.
- **Approximate movement metrics.** Distance, speed and sprint counts are uncalibrated heuristics; treat cross-duration magnitudes as indicative only.
- **Advisory-only workload output.** The workload/fatigue advisory is heuristic and explicitly *not* medical. In these single-block clips `distance_drop_pct` is a degenerate 100% for every track, so **advisory second-half conclusions are not scientifically meaningful and must not be compared as such.**

## Recommendation

**Ready with precautions for the 94-minute acceptance test.**

Justification: identical, verified configuration; a completed 46-minute real run on MPS with zero errors, no worker restart, no MPS→CPU fallback, and 7/7 artifact checksums intact; and stable detection density. The one caveat that makes this *with precautions* rather than unconditional is the **super-linear processing growth** observed from 15→45 min. Precautions:

- **Time budget:** the 45-min run took **1.375× real-time**, and per-minute cost is still rising, so a 94-min match should be budgeted at **≳ 1.4× real-time → plan for ~2.2–2.5 hours of wall-clock** and monitor that the rate does not degrade further (thermal throttling / tracker load are the likely causes).
- **Limits:** raise `FOOTBALLAI_MAX_UPLOAD_BYTES` and `FOOTBALLAI_MAX_VIDEO_DURATION_SECONDS` bounded just above the real file (a 94-min broadcast will be ~6 GB and ~5,640 s).
- **Disk:** ensure free space ≈ 2× the source video; the source is copied into the run input and dominates run-directory size (~6 GB for 94 min).
- **Stability:** confirm the device stays **MPS** (no silent CPU fallback), do not restart the worker mid-run, and do not launch a second heavy analysis concurrently.
- **Interpretation:** expect fragmentation and the insufficient-track percentage to rise further at 94 min; treat tracking and workload/fatigue outputs as advisory only, and prioritize a re-identification + pitch-homography upgrade before treating per-player analytics as authoritative.

