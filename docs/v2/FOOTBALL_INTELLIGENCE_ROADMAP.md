# Football intelligence roadmap (AI2–AI11)

**Product goal.** A coach uploads a full match of their team, an asynchronous
computer-vision analysis runs, and the coach explores individual and team
statistics, filters them by team, player, half, interval, event and pitch
region, opens the video moment behind any number, and asks a grounded AI
assistant questions that are answered only from that structured data.

This is the north star, not a description of today. It complements, and does
not replace, the historical portfolio roadmap in
[`../portfolio/AI_ENGINEERING_ROADMAP.md`](../portfolio/AI_ENGINEERING_ROADMAP.md)
(milestones M0–M5). AI1, AI2A and AI2B keep their established meanings.

## Where the product stands (audited 2026-11-06, `main` at `cd63073`)

| Capability | State |
|---|---|
| Upload, async analysis, immutable versioned runs | Implemented, verified |
| YOLOv8m detection cached once, ByteTrack replayable (AI1) | Implemented |
| Identity ground truth, cannot-links, metrics (AI2A) | Implemented, 18 s clip |
| Appearance benchmark (AI2B) | PR #37, unmerged; does not justify a resolver |
| Identity association, artifact, corrections (AI2C) | This milestone; `review_only` by default, automatic links experimental and provisional only |
| Pitch calibration / homography | Not implemented. V1 uses a constant 0.086 m/px scale |
| Ball, events, possession, passes, shots | Not implemented |
| Per-player statistics | V1 per-*fragment* statistics only |
| Evidence-grounded copilot, 5 read-only tools | Implemented, offline; real model never run live |

## Dependency graph

```text
AI1 tracklets ─┬─> AI2 identity ───────────────┐
               │      ^  metric-space gating   │
               └─> AI3 pitch calibration ──────┼─> AI5 game state ─> AI6 events
                         │                      │         │              │
AI4 ball tracking ───────┴──────────────────────┘         v              v
                                         AI7 player stats  AI8 team/tactical
                                                 └────────┬────────┘
                                                          v
                                 AI9 coach dashboard   AI11 multi-match comparison
                                                          v
                                          AI10 grounded coach assistant
```

Team-level spatial analytics (team heatmaps, width, length, compactness) need
AI3 and team assignment, but **not** persistent player identity. That is why
AI3 is the next milestone.

## Stages

Each stage: reusable code · missing · contracts · models/data · metrics ·
cost · risks · done when.

### AI2 — persistent player identity / ReID

- **Reuse:** AI1 readers; AI2A ground truth, cannot-links, team baseline; AI2C
  resolver, artifact, corrections, benchmark; AI2B encoder interface (if
  reviewed and merged separately).
- **Missing:** evidence that survives camera motion and long gaps: metric-space
  gating (needs AI3), jersey-number recognition, football-specific ReID, a
  role classifier (player/official/staff), a review UI for corrections at
  match scale.
- **Contracts:** `footballai.player-identities/v1`,
  `footballai.identity-corrections/v1`, `footballai.player-observation/v1`.
- **Data:** identity-disjoint labels across several matches and cameras;
  current labels are two short clips from two sources.
- **Metrics:** pairwise P/R/F1 with counts, false merges, contamination,
  cross-team and cannot-link violations, unresolved rate, fragments per
  identity; IDF1/HOTA once per-frame labels exist.
- **Cost:** resolver is negligible (≈1 s per full match); labelling dominates.
- **Risks:** false merges silently corrupt every per-player number.
- **Done when:** on held-out matches, zero certain false merges at a recall
  that makes per-player minutes meaningful (target to be set with the coach),
  and a reviewer can confirm a match's identities in bounded time.

### AI3 — pitch calibration and homography

- **Reuse:** AI1 sampled frames and boxes; `pitch_calibration` stage name in
  the analysis-run contract; observation contract slot `pitch_xy_m`.
- **Missing:** pitch keypoint/line detection, per-frame (or per-shot)
  homography with camera-motion handling, shot-cut detection, a calibration
  quality status, and a labelled keypoint benchmark.
- **Contract:** `footballai.pitch-calibration/v1` (per frame: homography or
  `UNCALIBRATED`, reprojection error, method, model checksum).
- **Data:** frames with annotated pitch keypoints across cameras/stadiums;
  licence must be checked before use.
- **Metrics:** reprojection error in metres on held-out keypoints, share of
  frames calibrated, failure rate on close-ups and replays.
- **Cost:** keypoint model inference per sampled frame (similar order to
  detection); bounded clips first, no new 90-minute run without approval.
- **Risks:** broadcast zoom/close-ups/replays; wrong calibration is worse than
  none, so `UNCALIBRATED` must be first-class.
- **Done when:** held-out reprojection error and coverage targets met, and the
  observation stream fills `pitch_xy_m` only for calibrated frames.

### AI4 — ball detection and temporal ball tracking

- **Missing:** a ball detector suited to small, blurred balls, temporal
  tracking with interpolation flags, in-play/out-of-play state.
- **Contract:** `footballai.ball-track/v1` (pixel and, when calibrated, pitch
  position; visibility; interpolated flag).
- **Metrics:** detection recall/precision at a pixel tolerance, track
  continuity, interpolation error on held-out labels.
- **Risks:** occlusion and motion blur; low frame rate (5 fps sampling) may be
  too coarse for passes.
- **Done when:** labelled clips meet recall targets and gaps are explicit.

### AI5 — structured game-state reconstruction

- **Inputs:** AI2 observations, AI3 calibration, AI4 ball.
- **Contract:** `footballai.game-state/v1`: per timestamp, players with
  team/identity status and pitch position, ball, possession candidate, phase,
  plus coverage.
- **Done when:** each state row carries the quality status of every input and
  no field is filled without its source.

### AI6 — football event detection

- **Events:** passes (attempted/completed), receptions, recoveries,
  interceptions, turnovers, shots; each with time, actors, pitch location,
  confidence, video reference.
- **Data:** event-labelled matches with a licence that allows training and
  evaluation.
- **Metrics:** event precision/recall at a time tolerance, actor accuracy.
- **Risks:** events need identity and ball quality; until then abstain.
- **Done when:** held-out event metrics published per event type, and
  unsupported types are reported as not available.

### AI7 — individual player statistics and heatmaps

- **Inputs:** observations with `confirmed` identity (provisional only by
  explicit opt-in), AI3 pitch positions.
- **Statistics:** minutes observed, distance, average/peak speed, sprints and
  high-intensity runs, heatmaps, zones, later event counts.
- **Contract:** a statistic envelope with value, unit, interval, scope,
  coverage, quality status, derivation, source checksums, video reference.
- **Done when:** physical metrics are validated against a reference (e.g.
  tracking-data provider or GPS) on held-out footage, with coverage shown.

### AI8 — team statistics and tactical analysis

- **Statistics:** team heatmaps and occupation, width/length/compactness,
  formation estimates, possession and passing (after AI6), phases, pressing,
  transitions, trends by half and interval.
- **Note:** spatial team metrics need AI3 plus team assignment only.
- **Done when:** each metric has a definition, a held-out check, and a
  quality status.

### AI9 — coach analytics dashboard

- Team/match selection, filters (team, player, half, interval, event, region),
  per-statistic coverage and quality badges, and a jump to the video moment.
- **Security:** real authentication and per-team authorisation before any
  multi-coach use; no URL-only isolation.

### AI10 — grounded AI coach assistant

```text
Coach question
  -> intent and match context
  -> authorised read-only analytics tools
  -> validated structured match statistics
  -> server-side evidence and citations
  -> grounded answer (+ optional player, heatmap or video reference)
```

Future tools, **none of which exist today**: `get_match_summary`,
`get_team_statistics`, `get_player_statistics`, `get_player_heatmap`,
`compare_players`, `get_match_events`, `get_tactical_patterns`,
`get_video_evidence`. Each is added only when its engine (AI7/AI8/AI6) and
contract tests exist. Until then the existing five read-only evidence tools
keep their behaviour, and questions about possession, passes or player
heatmaps must be answered with an explicit abstention. Live model calls need
their own approval.

### AI11 — multi-match comparison

- Cross-match player identity (squad registry, jersey numbers), normalised
  per-90 statistics with coverage, trend views.
- **Risk:** cross-match identity is a separate, harder problem than AI2.

## Next milestone

**AI3 — pitch calibration foundation**, as a bounded slice: a
`footballai.pitch-calibration/v1` contract with first-class `UNCALIBRATED`,
shot-cut detection, a small labelled keypoint benchmark on frames from the two
existing licensed local clips (pinned by video checksum, pixels not committed), and `pitch_xy_m` filled in the observation stream only
for frames that pass the quality gate. It unblocks team-level spatial
analytics directly and gives AI2 metric-space gating that camera pans cannot
break.
