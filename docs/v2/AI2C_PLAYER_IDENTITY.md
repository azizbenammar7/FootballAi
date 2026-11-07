# AI2C — persistent player identity foundation

AI2C adds the first identity-association layer on top of AI1 tracklets: a
versioned identity artifact, hard constraints that make impossible merges
unrepresentable, explicit `UNKNOWN`/abstention states, an append-only human
correction log, a leakage-aware benchmark, and the observation contract that
pitch calibration (AI3) and player statistics (AI7) will consume.

**It does not solve persistent player identity.** The benchmark below shows
that the automatic association it can justify today is precise on what it
links but links almost nothing, and that it made one certain false merge on a
held-out clip. Automatic identities are therefore `provisional` review
candidates. Only human-confirmed identities are released to downstream
consumers by default.

## Identifier model

| Concept | Example | Meaning |
|---|---|---|
| `detection_id` | `detection_000366_004` | One AI1 detector box in one sampled frame |
| `tracklet_id` | `tracklet_000014` | One temporary ByteTrack result. Never an identity |
| `player_identity_id` | `PID_000003` | One AI2C identity, scoped to one `footballai.player-identities/v1` artifact |

AI2A ground-truth labels such as `TEAM_A_04` remain evaluation labels. Resolver
identities live in the `PID_` namespace, so they cannot be mistaken for
ground truth. A `PID_` is not stable across artifacts or matches.

## Pipeline

```text
AI1 footballai.tracklets/v1  (or AI2A review samples, or legacy V1 tracks)
  -> per-tracklet summary: extent + first/last box   (O(tracklets) memory)
  -> eligibility: semantic class, human exclusions
  -> hard constraints, pairwise
       temporal overlap                     (physics; never overridable)
       explicit cannot-links                (AI2A evidence, human corrections)
       confident team conflict (>= 0.8)     (overridable by a human must-link)
       semantic-class mismatch
  -> human must-links first                 (rejected if they violate physics)
  -> automatic successor candidates
       strictly later start, gap <= max_gap_seconds
       no suspected shot cut in between     (>= 8 tracklet births in one frame)
       temporal_kinematic: foot-point displacement in body heights
         <= 1.0 + 3.0 * gap_seconds, and box-height ratio <= 1.6
  -> accept only mutual-best pairs with an ambiguity margin >= 0.2,
     otherwise record an abstention with up to three candidates
  -> merge in cost order, re-checking every constraint against every
     member of both groups (no transitive contradiction)
  -> confirmed | provisional | unresolved (UNKNOWN) | excluded
  -> footballai.player-identities/v1
```

Resolver modes:

| Mode | Status | Selected how |
|---|---|---|
| `review_only` | **Default.** No automatic association; only human corrections create identities | `ResolverConfig()`, or the CLI without `--mode` |
| `temporal_only` | Experimental | `ResolverConfig(mode=ResolverMode.TEMPORAL_ONLY)` or `--mode temporal_only` |
| `temporal_kinematic` | Experimental | `SELECTED_TEMPORAL_KINEMATIC` (the benchmark-selected parameters) or `--mode temporal_kinematic` |

Experimental modes never run implicitly. Their identities are always
`provisional`, and the CLI prints a warning when one is selected.

**Compatibility.** Before this safety gate, `ResolverConfig()` and `resolve`
without `--mode` meant `temporal_kinematic`. A caller that relied on that must
now pass the mode explicitly. Artifacts are unaffected: each records its full
configuration, including the mode, so the committed artifacts keep their
meaning and still reproduce byte for byte with the explicit mode. The numeric
defaults are unchanged and only take effect in an experimental mode.

`association_score` is `1 - cost` for automatic links. It orders candidates; it
is not a calibrated probability, and human links carry no score.

### Evidence channels

| Channel | State in AI2C |
|---|---|
| Temporal non-overlap, gap | Used |
| Image-space foot-point kinematics, box-scale continuity | Used, short gaps only |
| Track-birth burst as a shot-cut barrier | Used (fixed at 8, not tuned) |
| Team colour (AI2A) | Used only as a confident-conflict guard |
| AI2A explicit cannot-links | Used when supplied |
| Human corrections | Used |
| Appearance embeddings (AI2B, PR #37) | **Not used.** Unmerged; its own report concludes it does not justify a resolver; its embedding cache is not committed |
| Jersey number, pitch position, camera-motion compensation | **Not implemented** |

## `footballai.player-identities/v1`

Deterministic JSON (sorted keys, no wall-clock fields): the same inputs and
configuration give the same bytes. Sections:

- `source`: input kind, input SHA-256, AI1 tracklets/detections/video SHA-256;
- `resolver`: name, version, full configuration, channels used, channels
  unavailable with the reason, and score semantics;
- `evidence_inputs`: role, schema and SHA-256 of team evidence, cannot-links
  and the corrections log;
- `identities`: members, extent, observed seconds, observation count when
  known, team (`UNKNOWN` unless confident and consistent), minimum automatic
  score, link counts, evidence channels;
- `assignments`: one per input tracklet with status, reason and frame extent;
- `links`: automatic or human, with gap, normalised displacement, scale ratio
  and score, or the correction sequence;
- `abstentions`: refused associations and why (`ambiguous_successor`,
  `ambiguous_predecessor`, `not_mutual_best`, `group_constraint_conflict`);
- `corrections_applied`: the audit trail (sequence, action, outcome, reviewer,
  time);
- `summary` and `limitations`.

The reader re-validates the invariants: one assignment per tracklet, identity
membership agrees with assignments, no temporal overlap inside an identity,
link gaps agree with extents, links connect every member as a spanning tree,
and each identity's automatic/human link counts agree with the sources of its
actual links. Every human link must cite exactly one `applied` `must_link`
entry in `corrections_applied` for the same pair, and the artifact must record
the corrections log's SHA-256 in `evidence_inputs`. Relabelling an automatic
link as human to forge a `confirmed` identity, or an identity whose links do
not connect its tracklets, fails to load.

## Human corrections: `footballai.identity-corrections/v1`

An append-only log pinned to one tracklets checksum. Actions: `must_link`,
`cannot_link`, `exclude` (official, staff, spectator), and `revoke` (withdraws an
earlier entry without deleting it). Sequences must be contiguous from 1, every
entry needs a pseudonymous reviewer handle, a UTC timestamp and a reason, and
contradictory effective decisions are rejected. A `must_link` that violates
temporal overlap or an explicit cannot-link fails loudly. An identity whose
links are all human is `confirmed`. An identity with even one automatic link
stays `provisional`, so a human link never upgrades an automatic one.

`confirmed` means *asserted by a must-link in the recorded corrections log*,
nothing more. The reviewer field is a pseudonymous handle: it is not
authenticated, and it does not prove that a second, independent person reviewed
the evidence; any artifact built with corrections carries a limitation saying
so. The resolver refuses corrections unless `evidence_inputs` records the
SHA-256 of the log's canonical serialization (`corrections_to_json_bytes`), and
verifies it. The CLI accepts only canonical correction files, so the recorded
checksum is also the file's checksum. Human exclusions must likewise cite an
applied `exclude` entry. The log records the reviewer's reason but not the
frames or crops they looked at.

```json
{
  "schema": "footballai.identity-corrections/v1",
  "tracklets_sha256": "<AI1 tracklets SHA-256>",
  "entries": [
    {"sequence": 1, "action": "exclude", "tracklet_ids": ["tracklet_000001"],
     "reviewer": "analyst-1", "reviewed_at": "2026-11-06T10:00:00Z",
     "reason": "match official", "revokes_sequence": null}
  ]
}
```

## CLI (offline; no detector, video, or network)

```bash
PYTHONPATH=v2/src .venv-test/bin/python -m footballai_v2.execution.ai2c.cli resolve \
  --tracklets /path/to/tracklets.parquet --source-id my-match \
  --corrections /path/to/identity-corrections.json \
  --output /tmp/footballai-ai2c/player-identities.json
```

`--review-samples` resolves from AI2A review samples (exact extents and
endpoint boxes, used for the committed benchmarks whose Parquet caches are not
in Git). `--legacy-raw-tracks` is a diagnostic scale input. `--mode review_only`
is the default and disables automatic association; `--mode temporal_only` and
`--mode temporal_kinematic` are experimental opt-ins and print a warning.
`evaluate` scores an artifact against AI2A ground truth; `benchmark` runs the
protocol below. Outputs are never overwritten without `--overwrite`.

## Downstream contract: `footballai.player-observation/v1`

`iter_player_observations(tracklets, tracklets_sha256, identities,
identities_sha256)` streams one typed observation at a time: frame, time,
`tracklet_id`, `player_identity_id` or `None`, identity status and score, team,
pixel box, foot pixel point, tracking confidence, `pitch_xy_m=None`,
`pitch_calibration="uncalibrated"`, and both source checksums. It refuses an
identity artifact resolved from different tracklets, and it recomputes the
identity artifact's canonical SHA-256 (`player_identities_sha256`, equal to the
checksum of any file the resolver or CLI wrote) and refuses a mismatching
`identities_sha256`. The tracklets checksum is checked against the identity
artifact's provenance, not recomputed from the in-memory tracklets, so callers
must pass the checksum of the Parquet file they actually loaded.

- Default: only `confirmed` identities.
- `include_provisional=True` is an explicit opt-in; anything derived must carry
  the provisional status.
- `include_unresolved=True` yields tracklet-only diagnostics, never player data.
- Pitch coordinates cannot be set until AI3 supplies a validated calibration;
  the dataclass rejects them.

Future statistics (AI7/AI8) must carry: value and unit, time interval,
player/team scope, coverage (observed seconds vs. interval), quality status
derived from identity status and calibration status, derivation method and
version, source artifact checksums, and an optional video time reference. A
tracklet-only heatmap may exist only as a labelled provisional diagnostic.

## Benchmark

Committed evidence: [`evaluation/ai2c/`](../../evaluation/ai2c/README.md).

### Data

| Clip | Tracklets | Labels | Use |
|---|---:|---|---|
| `belgium-japan-wide-90s` (from PR #37 at `d108538`) | 448 | 40 confirmed player tracklets, 8 identities (4 per team), 8 tracklets without an identity (official, staff, or unidentifiable player) | Development: `TEAM_A_01, TEAM_A_03, TEAM_B_01, TEAM_B_03`. Held-out: `TEAM_A_02, TEAM_A_04, TEAM_B_02, TEAM_B_04` (AI2B's identity-disjoint split, reused) |
| `ai1-bounded-18s-validation` (AI2A, `main`) | 39 | 28 confirmed tracklets, 1 positive pair | Second held-out clip, never used for selection |
| Legacy V1 full match (`raw_tracks.parquet`) | 8,441 | none | Scale diagnostic only |

The resolver always runs over every tracklet of a clip; only the scored pairs
are restricted, so unlabelled distractors are present. No detector or tracker
was run.

### Protocol

`temporal_kinematic` was swept over 27 configurations (gap 1/2/3 s, speed
3/5/8 body heights/s, margin 0.1/0.2/0.3) on development identities only.
Rule: zero development false merges and zero labelled non-player
contamination, then most true-positive pairs, ties to the more conservative
setting. Selected: gap 1.0 s, speed 3.0, margin 0.2, exposed as
`SELECTED_TEMPORAL_KINEMATIC`, an explicit experimental opt-in, not the default. The
a-priori setting (2.0 s, 5.0, 0.2) is reported unchanged. All 27 settings had
zero development false merges.

### Results (measured)

Belgium–Japan, pairs among confirmed labelled tracklets:

| Method | Dev TP / FP / FN / TN | Held-out TP / FP / FN / TN | All-labelled TP / FP | Unresolved tracklets |
|---|---|---|---|---:|
| `review_only` (current state) | 0 / 0 / 46 / 164 | 0 / 0 / 41 / 130 | 0 / 0 | 448 / 448 |
| `temporal_only` (2 s) | 0 / 0 / 46 / 164 | 0 / 0 / 41 / 130 | 0 / 0 | 438 / 448 |
| `temporal_kinematic` a-priori | 2 / 0 / 44 / 164 | 0 / 0 / 41 / 130 | 2 / 0 | 377 / 448 |
| `temporal_kinematic` selected | 5 / 0 / 41 / 164 | **0 / 0 / 41 / 130** | 5 / 0 | 323 / 448 |

Selected configuration: development precision 1.0 (5/5), recall 0.109 (5/46);
held-out recall **0.0 (0/41)**, precision undefined (no predicted pairs).
Labelled fragments per identity fall from 5.25 to 4.25 on development and stay
at 4.75 on held-out. Zero cross-team, temporal-overlap, or labelled non-player
contamination on this clip.

AI2A 18-second clip (held out, 378 confirmed pairs, 1 positive):

| Method | TP / FP / FN / TN | Merges touching uncertain labels | Non-player contamination pairs |
|---|---|---:|---:|
| `review_only` | 0 / 0 / 1 / 377 | 0 | 0 |
| `temporal_only` | 0 / **1** / 1 / 376 | 0 | 1 |
| `temporal_kinematic` selected | 0 / **1** / 1 / 376 | 3 | 2 |

The certain false merge links `STAFF_03` to player `TEAM_A_08` across 0.8 s
(1.57 body heights, team colour below the 0.8 guard). Adding AI2A team
evidence and its 330 cannot-links did not change it. Two further links join
tracklets whose (uncertain) labels differ.

AI2B's numbers (appearance F1 0.4516 vs. 0.4121 temporal+team, held-out) come
from an all-pairs classifier, not a resolver, and need an embedding cache that
is not committed; they are cited, not reproduced.

### Scale (diagnostic proxy, no labels)

Legacy full match, 5,641 s, 8,441 tracklets, 160,327 observations: 0.96 s
resolve time and 38.5 MB peak traced Python memory in this session's cloud container,
1,025 automatic links, 887 provisional identities (median 2 tracklets, max 6),
units 8,441 → 7,416, 21.8% of observations inside a provisional identity, 52
suspected shot cuts. Given the AI2A false merge, these links are unverified.

### Not measured

HOTA, IDF1 and ID switches (need per-frame boxes for every visible player);
appearance channel (no cache); calibrated confidence; anything beyond two
short broadcast clips.

## Scientific risks (safety-gate review)

These were reviewed before AI3. None was fixed by retuning: no parameter was
changed and no held-out result was revisited.

- **The certain false merge (`STAFF_03` → `TEAM_A_08`, AI2A clip).**
  `tracklet_000008` → `tracklet_000043`, 0.8 s gap, 1.57 body heights, score
  0.54. Jersey-colour team evidence labels *both* tracklets `TEAM_A` (0.71 and
  0.77 confidence), so no team threshold could have blocked it. The team guard
  only fires on *disagreeing* confident labels. There is no role channel
  (player/staff/official), so a staff member near the touchline looks like a
  player to every automatic constraint. Mitigation: `review_only` default;
  staff and officials can only be removed by a human `exclude`.
- **Team and role constraints are conservative only for false merges they can
  see.** The team guard blocks confident cross-team merges. It cannot stop
  same-colour staff/player merges, and a weak or `UNKNOWN` label never blocks.
  A human must-link may override a confident team conflict, by design
  (team colour is not ground truth), so team quality ultimately rests on the
  reviewer.
- **Group-level transitive merging.** Every merge is re-checked against every
  member of both groups, so hard constraints cannot be broken by chaining. Soft
  errors still compound: AI2A identity `PID_000003` chains three tracklets
  (`TEAM_A_09` → uncertain `STAFF_11` → uncertain `TEAM_A_17`), with a weakest
  link score of 0.08, just inside the gate. `association_score_min` exposes
  this; it is not a probability.
- **Camera-cut detection** treats ≥ 8 track births in one sampled frame as a
  cut. It misses cuts to close-ups or replays with fewer people, gradual
  transitions and zooms; it can fire on crowded set-pieces. The threshold was
  fixed a priori and is not validated. Image-space kinematics also ignore
  pan/zoom, which is why gaps are capped at 1 s.
- **Missing AI2B appearance evidence.** PR #37 is unmerged, its embedding cache
  is not committed, and its own report did not justify a resolver. AI2C
  therefore has no channel that survives occlusion or camera motion.
- **The three-review-samples precondition.** Review-sample input assumes the
  default early/middle/late selection. A bundle built with `max_samples=2` is
  rejected (middle without late). A bundle built with `max_samples=1` cannot be
  detected from the format and would understate extents. Only
  `build_review_samples`, which always uses three, should produce resolver
  input. The authoritative input remains AI1 tracklets Parquet.
- **Evidence traceability of corrections.** A human link is traceable to one
  correction entry, its reason, reviewer handle and time, and the log's
  SHA-256. It is not traceable to the specific frames or crops the reviewer
  inspected, and the handle is not authenticated (see above).
- **Downstream exposure.** `iter_player_observations` yields only `confirmed`
  identities by default and verifies both checksums against the identity
  artifact. Provisional data needs `include_provisional=True` and keeps its
  status on every observation. No current consumer (dashboard, API, agent
  tools, statistics) reads AI2C artifacts, and the resolver is not wired into
  the `identity_resolution` run stage, so nothing can show a provisional
  identity as a verified player today. Any future consumer must keep these
  defaults.

## Conclusions

1. The hard-constraint, abstention, correction and artifact machinery works
   and is regression-tested.
2. Short-gap image-space stitching is not a persistent-identity method. On
   held-out identities it recovered nothing, and on another clip it confused a
   staff member with a nearby player. The positives that matter are
   separated by seconds of camera movement, not sub-second occlusions.
3. Automatic association is off by default (`review_only`). When explicitly
   enabled, its identities stay `provisional`; player statistics must not use
   them without review.
4. Identity needs evidence that survives camera motion and long gaps: pitch
   coordinates (AI3) for metric-space gating, plus jersey-number or
   football-specific re-identification evaluated across matches.
