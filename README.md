# Football Match Analysis Platform
**eSteps / Mitus.AI — AI Intern Technical Test**

End-to-end pipeline: detect & track players in a full 90-minute match →
compute per-player stats → derive fatigue & injury-risk indicators →
display in a Streamlit dashboard built for coaches and physios.

---

## Quick start

> **Python version:** use **3.11–3.13** in a virtual environment. The ML stack
> (torch / ultralytics / pyarrow / scipy) does not yet ship prebuilt wheels for
> 3.14, so a default `python3` that is 3.14 will try to build from source and
> hang. On macOS: `python3.13 -m venv .venv && source .venv/bin/activate`.

```bash
# 1. Install dependencies (Python 3.11–3.13)
pip install -r requirements.txt

# 2a. Run on a real match video
python pipeline/01_track.py --video data/raw/match.mp4   # ~30–45 min on M4
python pipeline/02_stats.py
python pipeline/03_fatigue.py

# 2b. OR generate synthetic demo data instantly (no video needed)
python scripts/gen_dummy_data.py

# 3. Launch dashboard
streamlit run dashboard/app.py
```

---

## Repository structure

```
├── pipeline/
│   ├── 01_track.py        # YOLOv8n + ByteTrack → raw_tracks.parquet
│   ├── 02_stats.py        # tracking data → player stats + heatmaps
│   └── 03_fatigue.py      # fatigue indicators + injury-risk scores
├── dashboard/
│   └── app.py             # Streamlit dashboard (reads precomputed files)
├── scripts/
│   └── gen_dummy_data.py  # generates synthetic match data for dashboard dev
├── data/
│   ├── raw/               # place your video here (gitignored)
│   └── processed/         # pipeline outputs (gitignored)
├── requirements.txt
└── README.md
```

---

## Architecture

```
match.mp4
   │
   ▼  01_track.py  (YOLOv8n, vid_stride=5, device=mps)
raw_tracks.parquet   [frame, time_sec, track_id, cx, cy, w, h, conf]
   │
   ▼  02_stats.py  (pandas + NumPy)
player_stats.parquet [track_id, block, half, distance_m, mean_speed, sprints, …]
player_summary.json  [per-player totals + heatmap + speed timeline]
   │
   ▼  03_fatigue.py  (scipy linregress + heuristic scoring)
risk_scores.json     [risk_score 0–100, risk_flag LOW/MEDIUM/HIGH, breakdown]
   │
   ▼  dashboard/app.py  (Streamlit + Plotly + Matplotlib)
```

The dashboard **never runs any ML**. It reads the three precomputed files,
which are typically < 5 MB total for a 90-minute match.

---

## Write-up: answers to §7

### 1. How did you get a full 90-minute match to process in reasonable time?

**Downsampling to 5 FPS** via `vid_stride=5` in Ultralytics.
A typical broadcast match at 25 FPS → 135 000 frames. At 5 FPS → 27 000 frames.
Combined with `yolov8n` (the smallest/fastest YOLO variant) on Apple MPS
(`device="mps"`), this reaches ~50–70 it/s on M4, giving a **~25–45 minute
wall-clock time** for the full pipeline.

Trade-offs:
- 5 FPS misses very short events (< 0.2 s). Fast turns look slightly slower.
- `yolov8n` has lower recall than `yolov8l`; occasional missed detections
  are acceptable given the test's stated tolerance for tracking imperfection.
- No multi-processing: a single Python process keeps implementation simple
  and avoids VRAM contention on shared MPS memory.

### 2. Pixel → real-world distance/speed conversion

A standard pitch is 105 m × 68 m. We measure (or estimate) the pitch width
in pixels from the video (`PITCH_PIXEL_WIDTH = 1820` for 1920-wide video)
and derive a single scale factor:

```
M_PER_PX = 105 / PITCH_PIXEL_WIDTH  ≈ 0.0577 m/px
```

Speed = Euclidean pixel displacement between consecutive frames × M_PER_PX ÷ Δt.

**Main error sources:**
1. Camera pan / zoom: the scale varies across the frame and over time. A homography
   (pitch-line detection → perspective transform) would fix this but takes 4–6 hours
   to implement reliably. For order-of-magnitude distance estimates it is not worth it.
2. Players near the edges of the frame are stretched (barrel distortion).
3. The bounding-box centre is the foot position; it jumps when a player crouches
   or is partially occluded.

**How we control error (`02_stats.py`, `compute_kinematics`):**
- **Position smoothing** — a rolling-median window (≈1 s) on the box centres
  removes per-frame jitter before any kinematics are computed.
- **Speed cap** — any step implying > 12 m/s is treated as an artefact and clipped.
- **Consistent distance** — distance is reconstructed from the *capped* speed, so
  speed and distance never disagree. (An earlier version capped speed but summed raw
  displacement, which inflated totals to 20–40 km/player.)
- **Gap handling** — if a track is lost for > 2 s (occlusion / re-acquisition), the
  re-appearance jump contributes zero distance instead of a teleport.

Typical resulting error: ±15–25% on total distance. Sufficient for relative
comparison across players within the same match.

### 3. Fatigue model

Four indicators, each scored 0–25, summed to a 0–100 risk score:

| Indicator | Calculation | Rationale |
|-----------|-------------|-----------|
| **Speed decay** | Linear slope of mean_speed per 15-min block | Bradley et al. (2010): ~12% speed drop across 90 min |
| **Sprint drop** | (H1 sprints − H2 sprints) / H1 sprints | Mohr et al. (2005): significant 2nd-half sprint reduction |
| **Distance drop** | (H1 dist − H2 dist) / H1 dist | Classic fatigue marker in GPS-based load monitoring |
| **HSR load** | Total distance relative to team 75th percentile | Malone et al. (2017): high running volume predicts next-day injury risk |

The **full 90-minute timeline is essential**: speed_decay and sprint_drop are
meaningless without at least 4–6 time-blocks to fit a trend. A 30-second clip
would produce a flat slope.

### 4. Injury-risk flag

Scores map directly:
- **LOW** < 40 — normal load, no significant fatigue signal
- **MEDIUM** 40–69 — one or two indicators elevated; monitor closely
- **HIGH** ≥ 70 — multiple fatigue signals; consider substitution
- **INSUFFICIENT** — track does not span both halves (see below)

The score is fully decomposable (the dashboard shows each sub-score). A coach
can understand at a glance *why* a player is flagged, which is more actionable
than a black-box probability.

**Handling ID switches honestly.** ByteTrack produces ID switches, so a 90-min
match yields many short-lived track IDs, not 22 clean players. The half-vs-half
indicators (sprint drop, distance drop) would otherwise flag every substituted
player or ID fragment as HIGH risk just because they only appear in one half.
We therefore only score fatigue for tracks present in **both halves** with
≥ 40 % match coverage; everything else is surfaced as **INSUFFICIENT** rather
than as a false positive. The dashboard defaults to hiding these partial tracks
but lets you toggle them back on.

### 5. Approximate time per part

| Part | Time |
|------|------|
| Reading test, architecture design | 1 h |
| Pipeline (01–03) | 4 h |
| Dashboard | 3 h |
| Dummy data + testing | 1 h |
| README + demo recording | 1 h |
| **Total** | **~10 h** |

### 6. Where AI tools were used

- **Claude (Cursor)**: architecture planning, boilerplate code, this README.
- **Self-written**: fatigue heuristic design, sprint/speed logic, calibration
  constants, debugging tracking edge cases.
- All generated code was reviewed, tested, and adjusted before committing.

### 7. With two more weeks I would build

1. **OpenCV homography** for frame-by-frame pitch calibration — removes the
   biggest accuracy bottleneck.
2. **Team colour clustering** (K-Means on jersey HSV) → auto-assign team A / B.
3. **Ball tracking** → possession stats, pass map, shooting zones.
4. **Event detection**: sprint starts, physical duels (proximity + deceleration),
   high-speed runs — pushes the dashboard from "load monitoring" to "game intelligence".
5. **Fine-tuned YOLOv8** on a football-specific dataset (SoccerNet-v2 has bounding-box
   labels) to drastically improve recall in crowded scenes.

---

## Robustness

- **Checkpointing** (`01_track.py`): detections are streamed to a CSV checkpoint
  every 3000 processed frames and only converted to the final parquet at the end.
  A crash (or Ctrl-C) at minute 85 keeps everything processed so far instead of
  losing the whole run.

## Known limitations

- Scale factor is approximate; absolute distance/speed values carry ±15–25% error.
- Track IDs are not persistent across the whole match (ByteTrack ID switches).
  Specifically, the 15-minute halftime break guarantees a complete ID wipe.
  **Solution:** The pipeline performs a "naive halftime stitch" by pairing the
  top 22 most active tracks from the 1st half with the top 22 from the 2nd half.
  Other short-lived fragments (< 1 min) are filtered out to keep the dashboard
  clean, while surviving partial tracks are scored as INSUFFICIENT.
- No team separation (all players pooled, including referee).
- Tested on MacBook Air M4 16 GB; CUDA users should change `device="mps"` to `device="cuda"`.

---

## Data sources

Any publicly available full-match football video works.
Options with open licensing:
- [SoccerNet](https://www.soccer-net.org/) — annotated broadcast matches
- [Roboflow Football datasets](https://universe.roboflow.com/search?q=football)
- YouTube tactical-cam uploads (check licence before use)
